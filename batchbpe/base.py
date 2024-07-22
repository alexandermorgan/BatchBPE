"""
Contains the base Tokenizer class and a few common helper functions, namely the
tokenizer save/load functionality, and the data import/save functionality.
"""
import unicodedata
import pandas as pd
from collections import Counter
from functools import lru_cache
import matplotlib.pyplot as plt
import requests
from datasets import load_dataset, IterableDataset, Dataset
from pyarrow import ChunkedArray
from joblib import Parallel, delayed, cpu_count
import time
import os
import json
import pdb
import regex as re



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

def _process_dicts(batch, compiled_pattern):
    counter = Counter()
    for item in batch:
        counter.update(re.findall(compiled_pattern, item['text']))
    return counter

def _process_string_scalar(batch, compiled_pattern):
    counter = Counter()
    for item in batch:
        counter.update(re.findall(compiled_pattern, item.as_py()))
    return counter

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.multiprocess = multiprocess
        if multiprocess:
            self.cpus = cpu_count()
        else:
            self.cpus = 1
        self.store_dict = store_dict
        self.stop_list_size = stop_list_size

    def _id_dict_to_list(self, ids):
        if self.stop_list_size > 0:
            # get twice as many to be sure to be able to get X chunks of length > 1
            top2X = ids.most_common(2*self.stop_list_size)
            # filter out single character chunks
            index = len(self.vocab)
            stop_index = index + self.stop_list_size
            for key, val in top2X:
                self.merges[key] = index
                self.vocab[index] = [*key.encode('utf-8')]
                index += 1
                if index == stop_index:
                    break

            return [([*key.encode('utf-8')], val) for key, val in ids.items() if key not in topX]
        return [([*key.encode('utf-8')], val) for key, val in ids.items()]

    def _import_data(self, data):
        # determine if `data` is a text as a string, a path to a file, a url to
        # a text document, a dictionary of datasets kwargs, or a list of any of
        # the above. Return a list of 2-tuples of bytes objects and their counts.
        print(f"Importing data of type {type(data)}")
        ids = Counter()
        default_kwargs = {'path': 'parquet', 'split': 'train'}
        if not isinstance(data, (list, tuple)):
            data = (data,)
        for item in data:
            # process dict or json file from previous data load
            if isinstance(item, dict) or (isinstance(item, str) and item.endswith('.json')):
                with open(item, 'r') as f:
                    _ids_dict = json.load(f)
                if not _ids_dict:
                    print(f'Dictionary loaded from {item} is empty.')
                    continue
                last_item = _ids_dict.popitem()
                if last_item[1] != 0:
                    print(f'Warning: the dictionary stored in {item} does not seem to have been saved by this \
                    tokenizer. Attempting to use it anyway...')
                    _ids_dict[last_item[0]] = last_item[1]
                elif last_item[0] != self.pattern:
                    print(f'Warning: the dictionary stored in {item} did not use the same split pattern. \
                    The dictionary was made using:\n\t{last_item[0]}\n\n\
                    The currnet split pattern is:\n\t{self.pattern}\n\n\
                    Proceeding to use the dictionary anyway...')
                ids.update(_ids_dict)
            elif isinstance(item, str):
                if item.startswith('https://') or item.startswith('http://'):
                    text = requests.get(item).text    # if it's a url, assume it's to a text file
                    ids.update(re.findall(self.compiled_pattern, text))
                elif os.path.isfile(item) and item.endswith('.txt'):
                    with open(item, 'r', encoding='utf-8') as f:
                        ids.update(re.findall(self.compiled_pattern, f.read()))
                else:   # assume the string is the text itself
                    ids.update(re.findall(self.compiled_pattern, item))
            elif isinstance(item, ChunkedArray):
                batch_size = len(item) // (self.cpus*2) or 1
                print(f'Processing in {self.cpus} batches of size {batch_size}')
                batches = [item[i:i + batch_size] for i in range(0, len(item), batch_size)]
                print(f'Processing {len(batches)} batches of size {batch_size}')
                # Process each batch in parallel
                results = Parallel(n_jobs=self.cpus)(delayed(_process_string_scalar)(batch, self.compiled_pattern) for batch in batches)
                for result in results:  # Aggregate results into one Counter
                    ids.update(result)
            elif isinstance(item, Dataset):
                # item = load_dataset(**item).data #.data['text']#[:400000] #[:345250] is one gigabyte
                print(f'Datasets branch, type of item: {type(item)}')
                print(f"Loading {item.size_in_bytes/1024/1024:.2f} MB dataset")
                if self.multiprocess and len(item) > 1:
                    batch_size = len(item) // (self.cpus*2) or 1
                    batches = [item[i:i + batch_size] for i in range(0, len(item), batch_size)]
                    print(f'Processing {len(batches)} batches of size {batch_size}')
                    # Process each batch in parallel
                    pdb.set_trace()
                    results = Parallel(n_jobs=self.cpus)(delayed(_process_dicts)(batch, self.compiled_pattern) for batch in batches)
                    for result in results:  # Aggregate results into one Counter
                        ids.update(result)
                else:
                    for _dict in item:
                        ids.update(re.findall(self.compiled_pattern, _dict['train']))
            elif isinstance(item, IterableDataset):
                print('Serially processing IterableDataset...')
                info = []
                for _dict in item:
                    ids.update(re.findall(self.compiled_pattern, _dict['text']))
                    info.append([len(ids), sum(ids.values())])

                self.df = pd.DataFrame(info, columns=['Unique Text Chunks', 'Total Text Chunks'])
                pdb.set_trace()

        if self.store_dict:
            ids[self.pattern] = 0   # store the pattern used to split the text as the last key
            formatted_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
            filename = f'{formatted_time}-dataset-dict.json'
            try:
                with open(filename, 'w') as f:
                    json.dump(ids, f)
                print(f"Stored dictionary of {len(ids)} keys to {filename}")
            except:
                print('Failed to store dictionary of dataset, continuing on to training...')
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
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
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
            assert version == "minbpe v1"
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
        if chunk in self.merges:
            return [self.merges[chunk]]
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
