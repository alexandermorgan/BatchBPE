"""
The QuickTokenizer is the same as the BatchTokenizer but combines the get_stats
and merge_batch functions into one call per batch, and does not safeguard
against the overcounting of pairs in chunks of text with repeated characters.
E.g., this tokenizer will count "aaaaa" as 4 "aa" pairs instead of 2 like the
BatchTokenizer.
"""
from .base import Tokenizer
from collections import defaultdict
from heapq import nlargest
import time


def get_stats(ids):
    """
    Given `ids`, a list of 2-tuples of iterables of ints and int values,
    returns a defaultdict with the counts of occurrences of all the consecutive
    pairs of integers within each bytes object, multiplied by the integer value
    associated with each key. This function does not count pairs between the last
    element of one key and the first element of the next key. The integer value
    associated with each key serves as a multiplier for the count of each pair
    within that object. This version of get_stats *does* count consecutive
    identical pairs within the same bytes object, so there is slight overcounting
    of these token pairs when the same token appears 3+ times in a row. If you
    want a version that avoids this overcounting, use the get_stats from the batch
    tokenizer.

    Example:
        get_stats({b'abc': 2, b'bcd': 1, b'eee': 1})
        -> defaultdict(<class 'int'>, {(97, 98): 1, (98, 99): 2, (99, 100): 1, (101, 101): 2})
    """
    counts = defaultdict(int)
    for chunk, num in ids:
        last_index = len(chunk) - 1
        i = 0
        while i < last_index:
            j = i + 1
            counts[(chunk[i], chunk[j])] += num
            i = j
    return counts

def merge_batch_get_stats(ids, pairs):
    counts = defaultdict(int)
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
            if i:
                counts[(chunk[i-1], chunk[i])] += num
            i = j
        if i and i == last_index:
            counts[(chunk[-2], chunk[i])] += num
    return counts

class QuickTokenizer(Tokenizer):
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0, freq_cutoff=1):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff)

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
        stats = get_stats(ids)   # stats are later updated by merge_batch_get_stats

        while merges_remaining > 0:
            seen_first = set()   # tokens seen in the first position in pairs
            seen_last = set()   # tokens seen in the last position in pairs
            pairs_to_merge = {}
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
            merges.update(pairs_to_merge)  # save the merges
            batch_count += 1
            if merges_remaining:   # no need to merge last batch
                stats = merge_batch_get_stats(ids, pairs_to_merge)   # replace pairs_to_merge keys in ids with their values
            if verbose:
                t2 = time.time()
                print(f"Batch {batch_count} merged {len(pairs_to_merge)} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}") # unique words: {len(ids)} processed words: {sum(ids.values())}")
                t1 = t2
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
