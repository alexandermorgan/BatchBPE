"""
Contains the BaseTokenizer class and a few common helper functions, namely the
tokenizer save/load functionality, the data import/save functionality, and the
encode/decode methods. To train a tokenizer, use the BatchTokenizer or
QuickTokenizer subclasses of this Tokenizer class.
"""
import unicodedata
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched, islice
from functools import lru_cache
import requests
import threading
from pyarrow import ChunkedArray, parquet
import time
import os
import regex as re
import csv

_tls = threading.local()  # thread-local compiled pattern cache
# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# These byte values can never be produced by UTF-8 encoding. Their token IDs
# are reclaimed by the first learned vocabulary entries.
DEAD_UTF8_BYTES = dict.fromkeys((0xC0, 0xC1, *range(0xF5, 0x100)))

# -----------------------------------------------------------------------------
# a few helper functions

def merge(ids, pair, idx, len_ids):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx.
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]

    - ids: mutable list of token ids to merge in place.
    - pair: (left, right) token ids to replace.
    - idx: new token id that replaces each occurrence of pair.
    - len_ids: current length of ids (avoids repeated len() calls).
    """
    i = 0
    while i + 1 < len_ids:
        j = i + 1
        if ids[i] == pair[0] and ids[j] == pair[1]:
            ids[i] = idx
            del ids[j]
            len_ids -= 1
        i = j
    return len_ids

def replace_control_characters(s: str) -> str:
    """Escape Unicode control characters so tokens print safely.

    - s: string that may contain control characters.
    """
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    """Pretty-print a token, escaping control characters.

    - t: raw token bytes.
    """
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def _process_text_batch(batch, pattern):
    """Split/count a batch of Python strings (worker helper).

    - batch: iterable of strings.
    - pattern: split regex, or None for open-field (whole doc = one chunk).
    """
    # Compile once per thread via thread-local storage. Sharing a single
    # compiled object across threads segfaults; recompiling every call with
    # cache_pattern=False is too slow for the GPT-4 pattern.
    # findall returns strings directly, skipping match object creation.
    t0 = time.monotonic()
    counter = Counter()
    if pattern is None:   # open-field: each document is a single chunk
        counter.update(batch)
        return counter, time.monotonic() - t0
    if getattr(_tls, 'pattern', None) != pattern:
        _tls.pattern = pattern
        _tls.compiled = re.compile(pattern)
    for text in batch:
        counter.update(_tls.compiled.findall(text))
    return counter, time.monotonic() - t0


def _process_string_scalar(batch, pattern):  # for pyarrow.ChunkedArray
    """Split/count an Arrow string batch (worker helper)."""
    return _process_text_batch((item.as_py() for item in batch), pattern)


def _looks_like_url(s: str) -> bool:
    """True only for short, single-line http(s) references meant to be fetched.

    Training documents often *begin* with a URL (or embed one in a larger
    string). Those must be treated as literal text, not downloaded — otherwise
    two train() calls can see different network responses and diverge.

    - s: candidate string to test.
    """
    if not (s.startswith("https://") or s.startswith("http://")):
        return False
    if len(s) > 512 or any(c in s for c in "\n\r\t "):
        return False
    return True

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """
    Base class for Tokenizers containing common supporting functionality,
    but not any actual tokenization logic.
    """
    def __init__(self, pattern=GPT4_SPLIT_PATTERN, multiprocess=True, store_dict=False,
                 stop_list_size=0, freq_cutoff=1, dedup: bool | None = None):
        """
        - pattern: split regex. Default GPT-4; None = open-field (whole doc = one chunk).
        - multiprocess: use multiple CPU cores when importing data.
        - store_dict: save the chunk Counter to a CSV after import (requires dedup).
        - stop_list_size: promote this many frequent multi-char chunks to vocab early.
        - freq_cutoff: drop chunks seen fewer times than this (1 = keep all with count >= 1).
        - dedup: Counter-dedup chunks on import. None = auto (True with a pattern, False open-field).
        """
        # Start with the 243 bytes that may occur in valid UTF-8. The remaining
        # 13 byte-ID slots are filled by the first learned vocabulary entries.
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self._vocab_id_cursor = 191
        self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern) if self.pattern is not None else None
        self.multiprocess = multiprocess
        if multiprocess:
            self._cpus = os.cpu_count() or 1
        else:
            self._cpus = 1
        self.store_dict = store_dict
        self.stop_list_size = stop_list_size
        self.stop_words = {}
        self.freq_cutoff = freq_cutoff
        self._set_dedup(dedup)

    def _set_dedup(self, dedup: bool | None = None) -> None:
        """Set whether to Counter-dedup chunks during import.

        None (default) means automatic: True when a split pattern is set, False
        for open-field (pattern=None), where full-document duplicates are rare.

        - dedup: True/False to force, or None for automatic.
        """
        self.dedup = self.pattern is not None if dedup is None else dedup

    def set_pattern(self, pattern: str | None, dedup: bool | None = None) -> None:
        """Set the chunk-splitting pattern and synchronize related state.

        - pattern: regex string, or None for open-field tokenization.
        - dedup: True/False to force, or None to choose based on the pattern.
        """
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern) if pattern is not None else None
        self._set_dedup(dedup)

    def _next_vocab_id(self) -> int:
        """Return and reserve the next learned-vocabulary token ID.

        New vocabulary entries use the 13 IDs that cannot occur as UTF-8 bytes
        before allocating IDs from 256 onward. The cursor only moves forward,
        avoiding a scan through every previously allocated ID.
        """
        while True:
            if self._vocab_id_cursor == 193:
                self._vocab_id_cursor = 245
            else:
                self._vocab_id_cursor += 1
            if (
                self._vocab_id_cursor not in self.vocab
                and self._vocab_id_cursor not in self.special_tokens.values()
            ):
                return self._vocab_id_cursor

    def _id_dict_to_list(self, ids, *, encode_with_vocab: bool = False):
        """
        Given a dictionary of token counts, return a list of lists where
        each list contains the count as the FIRST element followed by tokens.
        Stop words are separated if the user has set the stop_list_size class
        attribute to a positive integer.

        Fresh BPE (encode_with_vocab=False): each chunk becomes raw UTF-8 bytes.
        New split wave with an existing vocabulary (encode_with_vocab=True): each
        chunk is encoded with the current merges first, so pair counts reflect
        the stage-1 token ids rather than relearned byte pairs.

        - ids: Counter of chunk string -> count.
        - encode_with_vocab: encode chunks with current merges instead of raw bytes.
        """
        result = []
        str_encode = str.encode
        encode = self._encode_chunk_core if encode_with_vocab else (lambda k: str_encode(k, 'utf-8'))
        if self.stop_list_size:
            # get twice as many to be sure to be able to get X chunks of length > 1
            top2X = ids.most_common(2*self.stop_list_size)
            index = self._next_vocab_id()
            stop_words = {}
            for key, val in top2X:
                if len(key) > 1:
                    stop_words[key] = index
                    self.vocab[index] = str_encode(key, 'utf-8')
                    index = self._next_vocab_id()
                if len(stop_words) == self.stop_list_size:
                    break
            self.stop_words = stop_words
            
            while ids:
                key, val = ids.popitem()
                if key in self.stop_words or 1 < self.freq_cutoff > val:
                    continue
                # Count at the beginning, then tokens
                result.append(array('i', [val, *encode(key)]))
        else:
            result = [
                array('i', [val, *encode(key)])
                for key, val in ids.items()
                if not (1 < self.freq_cutoff > val)
            ]
        return result

    def _iter_chunk_texts(self, data):
        """Yield raw chunk strings without aggregating counts (no Counter dedup).

        Open-field: one yield per document. With a split pattern: one yield per
        regex match. CSV / pre-baked dicts yield each key once per unit count
        (a count of 3 yields the key three times) so pair totals stay correct.
        Generic iterables can yield strings, Hugging Face-style records with a
        ``text`` field, or batched records whose ``text`` field is iterable.

        - data: text, path(s), URL(s), dict(s), or list/iterator thereof.
        """
        if not isinstance(data, (list, tuple)):
            data = (data,)
        for item in data:
            if isinstance(item, str) and item.endswith('.csv'):
                with open(item, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip headers
                    for k, v in reader:
                        for _ in range(int(v)):
                            yield k
                continue
            if isinstance(item, str):
                if _looks_like_url(item):
                    item = requests.get(item).text
                elif os.path.isfile(item):
                    if item.endswith('.txt'):
                        with open(item, 'r', encoding='utf-8') as f:
                            if self.compiled_pattern is None:
                                yield f.read()
                            else:
                                for line in f:
                                    yield from (m.group() for m in re.finditer(self.compiled_pattern, line))
                        continue
                    if item.endswith('.parquet'):
                        pf = parquet.ParquetFile(item)
                        total_rows = pf.metadata.num_rows
                        row_batch_size = max(1, total_rows // (self._cpus * 40))
                        rows_done = 0
                        for rb in pf.iter_batches(columns=['text'], batch_size=row_batch_size):
                            col = rb.column('text')
                            for i in range(len(col)):
                                text = col[i].as_py()
                                if self.compiled_pattern is None:
                                    yield text
                                else:
                                    yield from self.compiled_pattern.findall(text)
                            rows_done += len(col)
                            print(f"  parquet {rows_done:,}/{total_rows:,} rows"
                                  f"  (streaming, no dedup)", flush=True)
                        continue
            if isinstance(item, dict):
                d = dict(item)
                last_key = next(reversed(d))
                last_val = d[last_key]
                if last_val == 0 and last_key == self.pattern:
                    del d[last_key]
                elif last_val == 0:
                    print(f'Warning: the dictionary or csv file passed did not use the same split pattern.')
                    del d[last_key]
                for k, v in d.items():
                    for _ in range(int(v)):
                        yield k
            elif isinstance(item, str):
                if self.compiled_pattern is None:
                    yield item
                else:
                    yield from (m.group() for m in re.finditer(self.compiled_pattern, item))
            elif isinstance(item, ChunkedArray):
                for i in range(len(item)):
                    text = item[i].as_py()
                    if self.compiled_pattern is None:
                        yield text
                    else:
                        yield from self.compiled_pattern.findall(text)
            elif isinstance(item, Iterable):
                for record in item:
                    texts = record['text'] if isinstance(record, Mapping) else record
                    if isinstance(texts, str):
                        texts = (texts,)
                    for text in texts:
                        if not isinstance(text, str):
                            raise TypeError(
                                'stream records must be strings or mappings with '
                                "a string or iterable 'text' field"
                            )
                        if self.compiled_pattern is None:
                            yield text
                        else:
                            yield from (m.group() for m in self.compiled_pattern.finditer(text))
            elif item is not None:
                print(f'Warning: unrecognised data type {type(item)}, skipping.')

    def _iter_chunk_arrays(self, data, *, encode_with_vocab: bool = False):
        """Yield `array('i', [1, *tokens])` for each chunk text (no dedup).

        - data: text, path(s), URL(s), dict(s), or list thereof.
        - encode_with_vocab: encode chunks with current merges instead of raw bytes.
        """
        if self.stop_list_size:
            raise ValueError("stop_list_size requires dedup=True")
        str_encode = str.encode
        encode = self._encode_chunk_core if encode_with_vocab else (lambda k: str_encode(k, 'utf-8'))
        for text in self._iter_chunk_texts(data):
            if 1 < self.freq_cutoff > 1:
                continue  # count is always 1; freq_cutoff>1 drops everything
            yield array('i', [1, *encode(text)])

    def _import_data(self, data, *, encode_with_vocab: bool = False) -> list:
        """
        Determine if `data` is a text as a string, a path to a file, a url to
        a text document, a dictionary of datasets kwargs, or a list of any of
        the above. Return a list of chunk arrays `[count, *token_ids]`.

        When self.dedup is false (the default for open-field), chunks are not
        aggregated in a Counter — each document/match becomes its own row.

        - data: text, path(s), URL(s), dict(s), or list thereof.
        - encode_with_vocab: encode chunks with current merges instead of raw bytes.
        """
        if not self.dedup:
            if self.store_dict:
                print('Warning: store_dict requires dedup=True; ignoring store_dict.')
            return list(self._iter_chunk_arrays(data, encode_with_vocab=encode_with_vocab))

        ids = Counter()
        if not isinstance(data, (list, tuple)):
            data = (data,)
        for item in data:
            # convert to ChunkedArray, dict, or str of text to parse
            if isinstance(item, str) and item.endswith('.csv'):   # csv file from previous data load
                with open(item, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip the headers
                    for k, v in reader:
                        ids[k] += int(v)
                    item = None  # skip the post-loop dict handling block
            elif isinstance(item, str):
                if _looks_like_url(item):
                    item = requests.get(item).text    # short URL path → fetch remote text
                elif os.path.isfile(item):
                    if item.endswith('.txt'):
                        with open(item, 'r', encoding='utf-8') as f:
                            if self.compiled_pattern is None:   # open-field: whole file is one chunk
                                ids[f.read()] += 1
                            else:
                                for line in f:
                                    ids.update(m.group() for m in re.finditer(self.compiled_pattern, line))
                        item = None  # skip the post-loop string handling block
                    elif item.endswith('.parquet'):
                        pf = parquet.ParquetFile(item)
                        total_rows = pf.metadata.num_rows
                        row_batch_size = max(1, total_rows // (self._cpus * 40))
                        rows_done = 0
                        batch_iter = pf.iter_batches(columns=['text'], batch_size=row_batch_size)
                        # future -> batch_rows; always keep _cpus futures in-flight.
                        # as_completed returns whichever finishes first so a free
                        # thread is never waiting on a slower sibling.
                        futures = {}

                        def _collect(future, batch_rows):
                            nonlocal rows_done
                            counter, batch_time = future.result()
                            ids.update(counter)
                            rows_done += batch_rows
                            print(f"  parquet {rows_done:,}/{total_rows:,} rows"
                                  f"  {batch_rows/batch_time:,.0f} rows/s"
                                  f"  {len(ids):,} unique tokens", flush=True)

                        with ProcessPoolExecutor(max_workers=self._cpus) as pool:
                            for rb in islice(batch_iter, self._cpus):
                                chunk = rb.column('text')
                                f = pool.submit(_process_string_scalar, chunk, self.pattern)
                                futures[f] = len(chunk)
                            for rb in batch_iter:
                                chunk = rb.column('text')
                                done = next(as_completed(futures))
                                _collect(done, futures.pop(done))
                                f = pool.submit(_process_string_scalar, chunk, self.pattern)
                                futures[f] = len(chunk)
                            for done in as_completed(futures):
                                _collect(done, futures.pop(done))
                        item = None  # skip the post-loop handling block
            # process data
            if isinstance(item, dict):
                last_item = item.popitem()
                if last_item[1] != 0:
                    print(f'Warning: the csv file or dictionary passed does not seem to have been made by this tokenizer.')
                    item[last_item[0]] = last_item[1]
                elif last_item[0] != self.pattern:
                    print(f'Warning: the dictionary or csv file passed did not use the same split pattern.')
                ids.update(item)
            elif isinstance(item, str):   # assume the string is the text itself
                if self.compiled_pattern is None:   # open-field: whole string is one chunk
                    ids[item] += 1
                else:
                    ids.update(m.group() for m in re.finditer(self.compiled_pattern, item))
            elif isinstance(item, ChunkedArray):
                batch_size = len(item) // (self._cpus*2) or 1
                batches = [*batched(item, batch_size)]
                print(f'Processing {len(batches)} batches of size {batch_size}')
                with ProcessPoolExecutor(max_workers=self._cpus) as pool:
                    for counter, _ in pool.map(
                        _process_string_scalar,
                        batches,
                        [self.pattern] * len(batches),
                    ):
                        ids.update(counter)
            elif isinstance(item, Iterable):
                # Consume generic streams (for example datasets.IterableDataset)
                # incrementally. The master Counter is the only full-corpus
                # structure; workers receive and return just small text batches.
                def iter_texts():
                    for record in item:
                        texts = record['text'] if isinstance(record, Mapping) else record
                        if isinstance(texts, str):
                            yield texts
                            continue
                        for text in texts:
                            if not isinstance(text, str):
                                raise TypeError(
                                    'stream records must be strings or mappings with '
                                    "a string or iterable 'text' field"
                                )
                            yield text

                text_batches = batched(iter_texts(), 32)
                if self._cpus == 1:
                    for batch in text_batches:
                        counter, _ = _process_text_batch(batch, self.pattern)
                        ids.update(counter)
                else:
                    futures = {}
                    with ProcessPoolExecutor(max_workers=self._cpus) as pool:
                        for batch in islice(text_batches, self._cpus):
                            future = pool.submit(_process_text_batch, batch, self.pattern)
                            futures[future] = None
                        for batch in text_batches:
                            done = next(as_completed(futures))
                            ids.update(done.result()[0])
                            del futures[done]
                            future = pool.submit(_process_text_batch, batch, self.pattern)
                            futures[future] = None
                        for done in as_completed(futures):
                            ids.update(done.result()[0])
            elif item is not None:
                print(f'Warning: unrecognised data type {type(item)}, skipping.')

        del item

        if self.store_dict:   # store dict compression of dataset to a csv file if requested
            ids[self.pattern] = 0   # store the pattern used to split the text as the last key
            formatted_time = time.strftime('%Y-%m-%d-%H_%M', time.localtime())
            filename = f'{formatted_time}-dataset-dict.csv'
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['text_chunk', 'count'])
                    for key, value in ids.items():
                        writer.writerow([key, value])
                print(f"Stored dictionary of {len(ids)} keys to {filename}")
            except:
                print('Failed to store dictionary of dataset.')
            del ids[self.pattern]   # remove the pattern key from the ids dict

        ids = self._id_dict_to_list(ids, encode_with_vocab=encode_with_vocab)
        return ids

    def train(self, text, vocab_size, verbose=False):
        """Train a vocabulary from text. Subclasses must implement this.

        - text: training corpus.
        - vocab_size: target vocabulary size.
        - verbose: print training progress.
        """
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {
            idx: bytes([idx])
            for idx in range(256)
            if idx not in DEAD_UTF8_BYTES
        }
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def register_special_tokens(self, special_tokens):
        """Register special tokens for encode/decode.

        - special_tokens: str -> int map, e.g. {"<|endoftext|>": 100257}.
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self._vocab_id_cursor = 191

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only

        - file_prefix: path prefix for the .model and .vocab files.
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("BatchBPE v1\n")
            # open-field (pattern is None) is written as an empty line
            f.write(f"{self.pattern if self.pattern is not None else ''}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for key in self.merges:
                if isinstance(key, tuple):
                    f.write(f"{key[0]} {key[1]}\n")
                else:
                    f.write(f"{key}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is a leaf token
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file.

        - model_file: path to a .model file previously written by save().
        """
        assert model_file.endswith(".model")
        # read the model file
        special_tokens = {}
        merge_pairs = []
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "BatchBPE v1"
            # read the pattern (an empty line means open-field / no splitting)
            self.pattern = f.readline().strip() or None
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merge_pairs.append((idx1, idx2))

        self.merges = {}
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
        self._vocab_id_cursor = 191
        for pair in merge_pairs:
            idx = self._next_vocab_id()
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.compiled_pattern = re.compile(self.pattern) if self.pattern is not None else None

    def decode(self, ids):
        """Decode token ids back to a Python string.

        - ids: list of integer token ids.
        """
        part_bytes = [self.vocab[idx] if idx in self.vocab
            else self.inverse_special_tokens[idx].encode("utf-8")
            for idx in ids] # raises KeyError if any idx is not a valid token
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk_core(self, chunk):
        """
        Given a chunk of text, return a list of integers representing the tokens.
        Uncached so it is safe to call during training (while self.merges is
        still growing between stages) without polluting the inference cache.

        - chunk: one text chunk (already split by the pattern, if any).
        """
        if chunk in self.stop_words:   # TODO: revisit this if statement
            return [self.stop_words[chunk]]
        # return the token chunk as a list of ints, similar to a bytes object
        chunk = [*chunk.encode("utf-8")]
        len_chunk = len(chunk)
        merges_get = self.merges.get
        while len_chunk >= 2:
            # find the pair with the lowest merge index
            low = 987654321
            for i in range(len_chunk - 1):
                current_pair = (chunk[i], chunk[i+1])
                new_val = merges_get(current_pair, 987654321)
                if new_val < low:
                    pair = current_pair
                    low = new_val
            if low == 987654321:   # no merges were found
                break   # nothing else can be merged
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            len_chunk = merge(chunk, pair, idx, len_chunk)
        return chunk   # list of ints

    @lru_cache(maxsize=131072)
    def _encode_chunk(self, chunk):
        """Cached wrapper around _encode_chunk_core, used by encode().

        - chunk: one text chunk (already split by the pattern, if any).
        """
        return self._encode_chunk_core(chunk)

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens.

        - text: string to encode.
        """
        if self.compiled_pattern is None:   # open-field: no splitting
            return self._encode_chunk(text)
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun

        - text: string to encode.
        - allowed_special: "all" | "none" | "none_raise" | set of special token strings.
          "none_raise" errors if any special token appears in text.
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:   # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # split on special tokens. Note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = f"({'|'.join([re.escape(k) for k in special])})"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            special_token = special.get(part)
            if special_token is None:   # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
            else:   # this is a special token, encode it separately as a special case
                ids.append(special_token)
        return ids
