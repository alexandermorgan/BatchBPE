"""
Contains the BaseTokenizer class and a few common helper functions, namely the
tokenizer save/load functionality, the data import/save functionality, and the
encode/decode methods. To train a tokenizer, use the BatchTokenizer or
QuickTokenizer subclasses of this Tokenizer class.
"""
import unicodedata
from collections import Counter
from functools import lru_cache
import requests
from datasets import load_dataset, IterableDataset, Dataset
from pyarrow import ChunkedArray
from joblib import Parallel, delayed, cpu_count
import time
import os
import regex as re
import csv


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# a few helper functions

def merge(ids, pair, idx, len_ids):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
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
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def _process_dicts(batch, compiled_pattern):   # for raw datasets.Dataset
    counter = Counter()
    for item in batch:
        counter.update(re.findall(compiled_pattern, item))
    return counter

def _process_string_scalar(batch, compiled_pattern):  # for pyarrow.ChunkedArray
    counter = Counter()
    for item in batch:
        counter.update(re.findall(compiled_pattern, item.as_py()))
    return counter

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """
    Base class for Tokenizers containing common supporting functionality,
    but not any actual tokenization logic.
    """
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0, freq_cutoff=1):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.multiprocess = multiprocess
        if multiprocess:
            self._cpus = cpu_count()
        else:
            self._cpus = 1
        self.store_dict = store_dict
        self.stop_list_size = stop_list_size
        self.stop_words = {}
        self.freq_cutoff = freq_cutoff

    def _id_dict_to_list(self, ids):
        """
        Given a dictionary of token counts, return a list of 2-tuples of bytes
        objects and their counts, with the stop words separated if the user has
        set the stop_list_size class attribute to a positive integer.
        """
        if self.stop_list_size:
            # get twice as many to be sure to be able to get X chunks of length > 1
            top2X = ids.most_common(2*self.stop_list_size)
            index = len(self.vocab)
            stop_index = index + self.stop_list_size
            stop_words = {}
            for key, val in top2X:
                if len(key) > 1: # and re.match(r'^ [A-Za-z\'’`]+$[A-Za-z]*', key):
                    stop_words[key] = index
                    self.vocab[index] = key.encode('utf-8')
                    index += 1
                if index == stop_index:
                    break
            self.stop_words = stop_words
            if self.freq_cutoff > 1:
                return [([*key.encode('utf-8')], val) for key, val in ids.items()
                        if (val >= self.freq_cutoff and key not in self.stop_words)]
            else:
                return [([*key.encode('utf-8')], val) for key, val in ids.items()
                        if key not in self.stop_words]
        else:   # self.stop_list_size == 0
            if self.freq_cutoff > 1:
                return [([*key.encode('utf-8')], val) for key, val in ids.items()
                        if val >= self.freq_cutoff]
            else:
                return [([*key.encode('utf-8')], val) for key, val in ids.items()]

    def _import_data(self, data) -> list[tuple[bytes, int]]:
        """
        Determine if `data` is a text as a string, a path to a file, a url to
        a text document, a dictionary of datasets kwargs, or a list of any of
        the above. Return a list of 2-tuples of bytes objects and their counts.
        """
        ids = Counter()
        if not isinstance(data, (list, tuple)):
            data = (data,)
        for item in data:
            # convert to ChunkedArray, dict, or str of text to parse
            if isinstance(item, Dataset):
                item = item.data['text']
            elif isinstance(item, str) and item.endswith('.csv'):   # csv file from previous data load
                with open(item, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    item = {k: int(v) for k, v in reader}
            elif isinstance(item, str):
                if item.startswith('https://') or item.startswith('http://'):
                    item = requests.get(item).text    # if it's a url, assume it's to a text file
                elif os.path.isfile(item):
                    if item.endswith('.txt'):
                        with open(item, 'r', encoding='utf-8') as f:
                            item = f.read()
                    elif item.endswith('.parquet'):
                        item = load_dataset('parquet', data_files=item).data['train'].flatten()[0]
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
                ids.update(re.findall(self.compiled_pattern, item))
            elif isinstance(item, ChunkedArray):
                batch_size = len(item) // (self._cpus*2) or 1
                batches = [item[i:i + batch_size] for i in range(0, len(item), batch_size)]
                print(f'Processing {len(batches)} batches of size {batch_size}')
                results = Parallel(n_jobs=self._cpus)(delayed(_process_string_scalar)(batch, self.compiled_pattern) for batch in batches)
                for result in results:  # Aggregate results into one Counter
                    ids.update(result)
            elif isinstance(item, IterableDataset):
                print('Serially processing IterableDataset...')
                for _dict in item:
                    ids.update(re.findall(self.compiled_pattern, _dict['text']))

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

        ids = self._id_dict_to_list(ids)
        return ids

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("BatchBPE v1\n")
            f.write(f"{self.pattern}\n")
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
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "BatchBPE v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = [self.vocab[idx] if idx in self.vocab
            else self.inverse_special_tokens[idx].encode("utf-8")
            for idx in ids] # raises KeyError if any idx is not a valid token
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @lru_cache(maxsize=131072)
    def _encode_chunk(self, chunk):
        """
        Given a chunk of text, return a list of integers representing the tokens.
        """
        if chunk in self.stop_words:   # TODO: revisit this if statement
            return [self.stop_words[chunk]]
        # return the token chunk as a list of ints, similar to a bytes object
        chunk = [*chunk.encode("utf-8")]
        len_chunk = len(chunk)
        while len_chunk >= 2:
            # find the pair with the lowest merge index
            low = 987654321
            for i in range(len_chunk - 1):
                current_pair = (chunk[i], chunk[i+1])
                new_val = self.merges.get(current_pair, 987654321)
                if new_val < low:
                    pair = current_pair
                    low = new_val
            if low == 987654321:   # no merges were found
                break   # nothing else can be merged
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            len_chunk = merge(chunk, pair, idx, len_chunk)
        return chunk   # list of ints

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
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
