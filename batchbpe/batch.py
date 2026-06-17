"""
Lightweight Byte Pair Encoding tokenizer. Merges are safely made in batches
along with other optimizations to be a practical tool for trying out new
tokenization strategies. Unlike the QuickTokenizer, the BatchTokenizer does not
combine the pair counting and token merging steps into the same function.
"""
from typing import Any
from .base import Tokenizer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from heapq import nlargest
import time


def _shards(seq, n):
    """Split `seq` into at most `n` contiguous slices for parallel workers."""
    size = (len(seq) + n - 1) // n or 1
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _merge_two(a, b):
    """Sum count dict `b` into `a` (the larger of the two) and return it."""
    for k, v in b.items():
        a[k] += v
    return a


def _combine_counts(parts, pool):
    """Sum a list of packed-int count dicts into one (the global pair stats).

    Reduced as a balanced binary tree: each wave merges disjoint pairs of dicts
    concurrently on `pool`, halving the count per wave, so the merge depth is
    ceil(log2(len(parts))) waves (e.g. 8 shards -> 3, 16 -> 4, 32 -> 5) instead
    of len(parts)-1 serial merges.
    """
    while len(parts) > 1:
        pairs = [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]
        merged = list[defaultdict[int, int]](pool.map(lambda ab: _merge_two(*ab), pairs))
        if len(parts) % 2:   # odd one out carries forward to the next wave
            merged.append(parts[-1])
        parts = merged
    return parts[0]


def get_stats(ids, mult):
    """
    Given `ids`, a list of lists where each list contains a count as the
    FIRST element followed by tokens, returns a defaultdict with the 
    counts of occurrences of all consecutive pairs of integers within each 
    list, multiplied by the count value. Consecutive identical pairs within 
    the same list are counted only once to avoid overcounting repeat characters.

    Pairs are keyed by the packed int `first*mult + last` instead of a
    `(first, last)` tuple, which avoids allocating a throwaway tuple for every
    adjacent pair in the hot loop. `mult` must exceed the largest token id
    (vocab_size) so that divmod recovers (first, last).

    Example (mult=1000):
        get_stats([[2, 97, 98, 99], [1, 98, 99, 100], [1, 101, 101, 101]], 1000)
        -> defaultdict(<class 'int'>, {97098: 2, 98099: 3, 99100: 1, 101101: 1})
    """
    counts = defaultdict(int)
    for chunk in ids:
        second_last_index = len(chunk) - 2  # second-to-last token index
        i = 1  # Start at index 1 (skip count)
        while i < second_last_index:
            j = i + 1
            counts[chunk[i] * mult + chunk[j]] += chunk[0]
            if chunk[i] == chunk[j] == chunk[i + 2]:
                i += 2  # skip the next token to avoid overcounting consecutive repeated pairs
            else:
                i = j
        if i == second_last_index:
            counts[chunk[i] * mult + chunk[i + 1]] += chunk[0]
    return counts


def merge_batch_and_get_stats(ids, pairs, mult):
    """
    Merge `pairs` into `ids` in place and return updated pair counts for the
    merged chunks. Counting uses the same consecutive-repeat guard as
    get_stats().

    Both `pairs` and the returned counts are keyed by the packed int
    `first*mult + last` instead of a `(first, last)` tuple, which avoids
    allocating a tuple for every pair lookup/count in the hot loop. `mult`
    must exceed the largest token id (vocab_size).

    The merge and the recount run as two separate passes (faster than a fused
    pass, as each is a tighter, more branch-predictable loop). Called per shard
    by train(); the recount is exactly get_stats() over the just-merged chunks.
    """
    pairs_get = pairs.get
    # --- merge pass: apply pairs to every chunk in place ---
    for chunk in ids:
        last_index = len(chunk) - 1
        i = 1
        while i < last_index:
            j = i + 1
            token = pairs_get(chunk[i] * mult + chunk[j])
            if token is not None:
                chunk[i] = token
                del chunk[j]
                last_index -= 1
            i = j
    # --- count pass: recompute pair stats over the merged chunks ---
    counts = defaultdict(int)
    for chunk in ids:
        num = chunk[0]
        second_last_index = len(chunk) - 2
        i = 1
        while i < second_last_index:
            j = i + 1
            counts[chunk[i] * mult + chunk[j]] += num
            if chunk[i] == chunk[j] == chunk[i + 2]:
                i += 2  # skip the next token to avoid overcounting consecutive repeated pairs
            else:
                i = j
        if i == second_last_index:
            counts[chunk[i] * mult + chunk[i + 1]] += num
    return counts

class BatchTokenizer(Tokenizer):
    def __init__(self, pattern=None, multiprocess=True, store_dict=False, stop_list_size=0, freq_cutoff=0):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff)

    def train(self, data, vocab_size, cap_divisor=2, max_batch_size=0, verbose=False,
              progress_callback=None):
        """
        Trains the tokenizer on the given data to the specified vocab_size. You
        probably don't want to change the cap_divisor or max_batch_size defaults.
        progress_callback: optional callable(rows_done, total_rows, batch_rows,
                           batch_time, elapsed, unique_tokens) called after each
                           row batch during parquet ingestion.
        """
        t0 = time.time()
        ids = self._import_data(data, progress_callback=progress_callback)
        t1 = time.time()
        print(f'Time spent loading data: {t1-t0:.2f}s')

        merges = self.merges   # {(int, int): int} -> token pair to new token
        vocab = self.vocab   # {int: bytes} -> token to its bytes representation
        batch_count = 0
        curr_vocab_size = len(vocab) + len(self.special_tokens)
        num_merges = vocab_size - curr_vocab_size
        merges_remaining = num_merges
        if max_batch_size < 1:
            max_batch_size = num_merges

        # Pairs in the hot stats/merge dicts are packed into a single int
        # (first*mult + last) to avoid allocating a tuple per adjacent pair.
        # mult exceeds every token id, so divmod recovers (first, last).
        mult = vocab_size
        n = self._cpus
        seen_first = set[int]()   # tokens seen in the first position in pairs
        seen_last = set[int]()   # tokens seen in the last position in pairs
        add_first = seen_first.add
        add_last = seen_last.add
        pairs_to_merge = {}

        # The merge/count work is sharded across threads: each thread processes a
        # disjoint slice of `ids` (distinct chunk objects, so the in-place merge is
        # safe under free-threaded Python) and returns a local count dict, which the
        # main thread sums into the global stats. The pool is reused across batches.
        with ThreadPoolExecutor(max_workers=n) as pool:
            stats = _combine_counts(list(pool.map(
                lambda shard: get_stats(shard, mult), _shards(ids, n))), pool)

            while merges_remaining > 0:
                num_pairs_to_search = min(merges_remaining//cap_divisor, curr_vocab_size, max_batch_size) or 1
                top_pairs = nlargest(num_pairs_to_search, stats, key=stats.get)
                for packed in top_pairs:  # pairs are packed ints: first*mult + last
                    first, last = divmod(packed, mult)
                    unsafe = first in seen_last or last in seen_first   # unsafe merge
                    add_first(first)
                    add_last(last)
                    if unsafe:
                        continue # skip this pair but keep looking for safe merges in top_pairs
                    pairs_to_merge[packed] = curr_vocab_size
                    merges[(first, last)] = curr_vocab_size  # model keeps tuple keys
                    vocab[curr_vocab_size] = vocab[first] + vocab[last]
                    curr_vocab_size += 1
                merges_remaining -= (num_pairs_to_merge := len(pairs_to_merge))
                batch_count += 1
                if merges_remaining:   # no need to merge last batch
                    # remove chunks that have solidified into a single token
                    if batch_count % 90 == 0:
                        ids = [chunk for chunk in ids if len(chunk) > 2]
                    stats = _combine_counts(list[defaultdict[int, int]](pool.map(
                        lambda shard: merge_batch_and_get_stats(shard, pairs_to_merge, mult),
                        _shards(ids, n))), pool)
                    seen_first.clear()
                    seen_last.clear()
                    pairs_to_merge.clear()

                if verbose:
                    t2 = time.time()
                    print(f"Batch {batch_count} merged {num_pairs_to_merge} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}")
                    t1 = t2
