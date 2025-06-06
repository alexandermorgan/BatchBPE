"""
Lightweight Byte Pair Encoding tokenizer. Merges are safely made in batches
along with other optimizations to be a practical tool for trying out new
tokenization strategies. Unlike the QuickTokenizer, the BatchTokenizer does not
combine the pair counting and token merging steps into the same function.
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
    within that object. Consecutive identical pairs within the same bytes object
    are counted only once to avoid overcounting repeat characters.

    Example:
        get_stats([([97, 98, 99], 2), ([98, 99, 100], 1), ([101, 101, 101], 1)])
        -> defaultdict(<class 'int'>, {(97, 98): 1, (98, 99): 2, (99, 100): 1, (101, 101): 1})
    """
    counts = defaultdict(int)
    for chunk, num in ids:
        last_index = len(chunk) - 1
        i = 0
        while i < last_index:
            j = i + 1
            counts[(chunk[i], chunk[j])] += num
            if chunk[i] == chunk[j] and j+1 <= last_index and chunk[i] == chunk[j+1]:
                i += 2  # skip the next token to avoid overcounting consecutive repeated pairs
            else:
                i = j
    return counts

def merge_batch(ids, pairs):
    """
    Given `ids`, a list of 2-tuples of iterables of ints and int values, and
    `pairs`, a dictionary of 2-tuples of ints and int values, returns a list of
    2-tuples of iterables of ints and int values with the pairs merged.
    """
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
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0, freq_cutoff=0):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff)

    def train(self, data, vocab_size, cap_divisor=2, max_batch_size=0, verbose=False):
        """
        Trains the tokenizer on the given data to the specified vocab_size. You
        probably don't want to change the cap_divisor or max_batch_size defaults.
        """
        t0 = time.time()
        ids = self._import_data(data)   # [(list_of_int_tokens, int)] -> text chunks and their counts
        t1 = time.time()
        print(f'Time spent loading data: {t1-t0:.2f}')

        merges = self.merges   # {(int, int): int} -> token pair to new token
        vocab = self.vocab   # {int: bytes} -> token to its bytes representation
        batch_count = 0
        curr_vocab_size = len(vocab) + len(self.special_tokens)
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
            merges.update(pairs_to_merge)  # save the merges
            batch_count += 1
            if merges_remaining:   # no need to merge last batch
                merge_batch(ids, pairs_to_merge)   # replace pairs_to_merge keys in ids with their values
            if verbose:
                t2 = time.time()
                print(f"Batch {batch_count} merged {len(pairs_to_merge)} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}")
                t1 = t2
