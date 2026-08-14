"""
Lightweight Byte Pair Encoding tokenizer. Merges are safely made in batches
along with other optimizations to be a practical tool for trying out new
tokenization strategies. Unlike the QuickTokenizer, the BatchTokenizer does not
combine the pair counting and token merging steps into the same function.
"""
from .base import GPT4_SPLIT_PATTERN, Tokenizer
from .corpus import Corpus, DiskCorpus, RamCorpus
from heapq import nlargest
import time


class BatchTokenizer(Tokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN, multiprocess: bool = True, store_dict: bool = False,
                 stop_list_size: int = 0, freq_cutoff: int = 0, dedup: bool | None = None) -> None:
        """
        - pattern: split regex. Default GPT-4; None = open-field (whole doc = one chunk).
        - multiprocess: use multiple CPU cores when importing data.
        - store_dict: save the chunk Counter to a CSV after import (requires dedup).
        - stop_list_size: promote this many frequent multi-char chunks to vocab early.
        - freq_cutoff: drop chunks seen fewer times than this (0 = keep all).
        - dedup: Counter-dedup chunks on import. None = auto (True with a pattern, False open-field).
        """
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff, dedup)
        self._corpus_ids = None
        self._corpus_pattern = None

    def train(self, data, vocab_size: int, cap_divisor: int = 2,
              max_batch_size: int = 0, backend: str = "ram", work_dir: str | None = None,
              memory_efficient: bool = False, verbose: bool = False,
              records_per_shard: int | None = None) -> None:
        """
        Trains the tokenizer on the given data to the specified vocab_size. You
        probably don't want to change the cap_divisor or max_batch_size defaults.

        - data: text, path(s), URL(s), list thereof, or a stream of text records.
        - vocab_size: target number of active vocabulary entries, including the
          initial 243 UTF-8 byte tokens.
        - cap_divisor: divides remaining merges to size each batch (default 2).
        - max_batch_size: hard cap on merges per batch; 0 = no cap beyond remaining.
        - backend: "ram" (in-memory) or "disk" (sharded corpus for large data).
        - work_dir: working directory for the "disk" backend; temp dir if omitted.
        - memory_efficient: cap pair-count dicts near vocab_size (less RAM, may differ slightly).
        - records_per_shard: disk-backend chunk buffer size; lower values use less
          import RAM but create more shard files. Required for bounded streaming
          input: use backend="disk" and dedup=False.
        - verbose: print per-batch timing during the merge loop.

        Calling train() again with the same split pattern and backend="ram"
        reuses the in-memory tokenized corpus from the previous run (no
        re-import). Changing the pattern starts a new wave: the corpus is
        reloaded and each chunk is encoded with the current vocabulary before
        merging. backend="disk" always re-imports (and clears the RAM corpus
        cache) so merge-loop state stays consistent with on-disk shards.
        When data is a list, backend="disk" also clears that list after writing
        shards so document strings are not retained through the merge loop.
        """
        if backend not in ("ram", "disk"):
            raise ValueError(f"backend must be 'ram' or 'disk', got {backend!r}")
        if records_per_shard is not None and records_per_shard < 1:
            raise ValueError("records_per_shard must be at least 1")

        encode_with_vocab = bool(self.merges)
        max_stats_size = vocab_size*5 if memory_efficient else 0
        t0 = time.time()

        if backend == "disk":
            self._corpus_ids = None
            self._corpus_pattern = None
            if self.dedup:
                ids = self._import_data(data, encode_with_vocab=encode_with_vocab)
                t1 = time.time()
                print(f'Time spent loading data: {t1-t0:.2f}s')
                with DiskCorpus(ids, self._cpus, work_dir,
                                max_stats_size=max_stats_size) as corpus:
                    del ids  # shards are on disk; free the in-RAM chunk list
                    if isinstance(data, list):
                        data.clear()  # free source strings before the merge loop
                    self._build_merges(corpus, vocab_size, cap_divisor, max_batch_size, t1, verbose)
            else:
                # Stream chunks straight to shards — never build a full Counter
                # or in-RAM list of arrays.
                corpus_kwargs = {"max_stats_size": max_stats_size}
                if records_per_shard is not None:
                    corpus_kwargs["records_per_shard"] = records_per_shard
                corpus = DiskCorpus.from_chunk_iter(
                    self._iter_chunk_arrays(data, encode_with_vocab=encode_with_vocab),
                    self._cpus, work_dir, **corpus_kwargs)
                if isinstance(data, list):
                    data.clear()  # free source strings before the merge loop
                t1 = time.time()
                print(f'Time spent loading data: {t1-t0:.2f}s')
                with corpus:
                    self._build_merges(corpus, vocab_size, cap_divisor, max_batch_size, t1, verbose)
            return

        same_wave = (
            self._corpus_ids is not None
            and self.pattern == self._corpus_pattern
        )
        if same_wave:
            ids = self._corpus_ids
            print('Reusing in-memory corpus (same split pattern).')
        else:
            ids = self._import_data(data, encode_with_vocab=encode_with_vocab)
            self._corpus_ids = ids
            self._corpus_pattern = self.pattern
        t1 = time.time()
        if not same_wave:
            print(f'Time spent loading data: {t1-t0:.2f}s')

        with RamCorpus(ids, self._cpus, max_stats_size=max_stats_size) as corpus:
            self._build_merges(corpus, vocab_size, cap_divisor, max_batch_size, t1, verbose)

    def _build_merges(self, corpus: Corpus, vocab_size: int, cap_divisor: int,
                      max_batch_size: int, t1: float, verbose: bool) -> None:
        """
        Run the batched BPE merge loop against any Corpus backend. This is the
        backend-agnostic core: it only asks `corpus` for the initial pair counts
        and for a merge-batch-then-recount each wave; where the chunk data lives
        and how the work is parallelized is the backend's concern.

        - corpus: Corpus backend holding tokenized chunks (RAM or disk).
        - vocab_size: target vocabulary size to grow toward.
        - cap_divisor: divides remaining merges to size each batch.
        - max_batch_size: hard cap on merges per batch; <1 means use all remaining.
        - t1: start timestamp for verbose batch timing.
        - verbose: print per-batch progress.
        """
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
        seen_first = set[int]()   # tokens seen in the first position in pairs
        seen_last = set[int]()   # tokens seen in the last position in pairs
        add_first = seen_first.add
        add_last = seen_last.add
        pairs_to_merge = {}

        stats = corpus.initial_stats(mult)
        while merges_remaining > 0:
            num_pairs_to_search = min(merges_remaining//cap_divisor, curr_vocab_size, max_batch_size) or 1
            # Tie-break on the packed pair id so ram/disk agree even when shard
            # layouts and stats dict insertion orders differ.
            top_pairs = nlargest(num_pairs_to_search, stats,
                                 key=lambda p: (stats[p], p))
            for packed in top_pairs:  # pairs are packed ints: first*mult + last
                first, last = divmod(packed, mult)
                unsafe = first in seen_last or last in seen_first   # unsafe merge
                add_first(first)
                add_last(last)
                if unsafe:
                    continue # skip this pair but keep looking for safe merges in top_pairs
                idx = self._next_vocab_id()
                pairs_to_merge[packed] = idx
                merges[(first, last)] = idx  # model keeps tuple keys
                vocab[idx] = vocab[first] + vocab[last]
                curr_vocab_size += 1
            merges_remaining -= (num_pairs_to_merge := len(pairs_to_merge))
            batch_count += 1
            if not num_pairs_to_merge:
                raise RuntimeError(
                    "BPE merge loop made no progress: empty stats or no safe pairs to merge "
                    f"(stats={len(stats)}, merges_remaining={merges_remaining}). "
                    "Often caused by an empty training corpus.")
            stats = corpus.merge_and_recount(pairs_to_merge, mult)
            seen_first.clear()
            seen_last.clear()
            pairs_to_merge.clear()

            if verbose:
                t2 = time.time()
                print(f"Batch {batch_count} merged {num_pairs_to_merge} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}")
                t1 = t2
