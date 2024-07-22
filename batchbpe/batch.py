"""
Lightweight Byte Pair Encoding tokenizer. Merges are safely made in batches
along with other optimizations to be a practical tool for trying out new
tokenization strategies.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

"""
from .base import Tokenizer
from collections import defaultdict
from heapq import nlargest
import time
import pdb


def get_stats(ids):
    """
    Given `ids`, a list of 2-tuples of iterables of ints and int values,
    returns a defaultdict with the counts of occurrences of all the consecutive
    pairs of integers within each bytes object, multiplied by the integer value
    associated with each key. This function does not count pairs between the last
    element of one key the first element of the next key. The integer value
    associated with each key serves as a multiplier for the count of each pair
    within that object. Consecutive identical pairs within the same bytes object
    are counted only once to avoid overcounting repeat characters. This will
    multiprocess using the number of cpus specified.

    Example:
        get_stats({b'abc': 2, b'bcd': 1, b'eee': 1})
        -> defaultdict(<class 'int'>, {(97, 98): 1, (98, 99): 2, (99, 100): 1, (101, 101): 1})
    """
    counts = defaultdict(int)
    for chunk, num in ids:
        last_index = len(chunk) - 1
        i = 0
        prev_pair = ''
        while i < last_index:
            j = i + 1
            this_pair = (chunk[i], chunk[j])
            if this_pair != prev_pair:
                counts[this_pair] += num
                prev_pair = this_pair
            else:
                prev_pair = ''
            i = j
    return counts

def merge_batch(ids, pairs):
    for chunk, num in ids:
        last_index = len(chunk) - 1
        i = 0
        while i < last_index:
            j = i + 1
            token = pairs.get((chunk[i], chunk[j]))
            if token is not None:
                chunk[i] = token
                del chunk[j]
                last_index -= 1
            i = j

class BatchTokenizer(Tokenizer):
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__(pattern, multiprocess, store_dict)
        self.stop_list_size = stop_list_size

    def train(self, data, vocab_size, cap_divisor=2, max_batch_size=0, verbose=False):
        t0 = time.time()
        ids = self._import_data(data)   # [(bytes, int)] -> text chunks and their counts
        t1 = time.time()
        print(f'Time spent loading data: {t1-t0:.2f}')

        merges = self.merges   # {(int, int): int} -> token pair to new token
        vocab = self.vocab   # {int: bytes} -> token to its bytes representation
        batch_count = 0
        curr_vocab_size = len(vocab)
        num_merges = vocab_size - curr_vocab_size
        merges_remaining = num_merges
        if max_batch_size < 1:
            max_batch_size = num_merges

        while merges_remaining > 0:
            seen_first = set()   # tokens seen in the first position in pairs
            seen_last = set()   # tokens seen in the last position in pairs
            pairs_to_merge = {}
            stats = get_stats(ids)   # count the number of times every consecutive pair appears
            num_pairs_to_search = min(merges_remaining//cap_divisor, len(vocab), max_batch_size) or 1
            top_pairs = nlargest(num_pairs_to_search, stats, key=stats.get)
            for first, last in top_pairs:  # pairs are (first, last) tuples
                if first in seen_last or last in seen_first:   # unsafe merge
                    seen_first.add(first)
                    seen_last.add(last)
                    continue # skip this pair but keep looking for safe merges in top_pairs
                seen_first.add(first)
                seen_last.add(last)
                pairs_to_merge[(first, last)] = curr_vocab_size
                vocab[curr_vocab_size] = vocab[first] + vocab[last]
                curr_vocab_size += 1
            merges_remaining -= len(pairs_to_merge)
            # replace all occurrences in ids of pairs_to_merge keys with their values
            merge_batch(ids, pairs_to_merge)
            merges.update(pairs_to_merge)  # save the merges
            batch_count += 1
            if verbose:
                t2 = time.time()
                print(f"Batch {batch_count} merged {len(pairs_to_merge)} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}") # unique words: {len(ids)} processed words: {sum(ids.values())}")
                t1 = t2
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
