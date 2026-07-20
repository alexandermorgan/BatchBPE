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
    def __init__(self, pattern=GPT4_SPLIT_PATTERN, multiprocess: bool = True, store_dict: bool = False, stop_list_size: int = 0, freq_cutoff: int = 0) -> None:
        """
        - pattern: split pattern. Omit for the GPT-4 default; pass an explicit
          regex string to override it; pass None for open-field merges (no
          splitting, the whole document is one chunk).
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff)
        # In-memory chunk list from the last import; reused across train() calls
        # when the split pattern is unchanged so same-wave resume (e.g. 40k merges,
        # save, 5k more) skips re-reading and re-encoding the corpus entirely.
        # Only used by backend="ram"; disk always re-imports (encode_with_vocab
        # when merges exist) so a stale post-merge RAM cache cannot poison a
        # later run.
        self._corpus_ids = None
        self._corpus_pattern = None

    def train(self, data: str | list[str], vocab_size: int, cap_divisor: int = 2,
              max_batch_size: int = 0, backend: str = "ram", work_dir: str | None = None,
              verbose: bool = False) -> None:
        """
        Trains the tokenizer on the given data to the specified vocab_size. You
        probably don't want to change the cap_divisor or max_batch_size defaults.

        - backend: "ram" (default, the typical case) keeps the deduplicated
          corpus in memory; "disk" streams a sharded corpus from disk for
          SuperBPE-style continued training on datasets too large for RAM.
        - work_dir: working directory for the "disk" backend. Defaults to a
          temporary directory when not supplied. Ignored by the "ram" backend.

        Calling train() again with the same split pattern and backend="ram"
        reuses the in-memory tokenized corpus from the previous run (no
        re-import). Changing the pattern starts a new wave: the corpus is
        reloaded and each chunk is encoded with the current vocabulary before
        merging. backend="disk" always re-imports (and clears the RAM corpus
        cache) so merge-loop state stays consistent with on-disk shards.
        """
        if backend not in ("ram", "disk"):
            raise ValueError(f"backend must be 'ram' or 'disk', got {backend!r}")

        t0 = time.time()
        if backend == "disk":
            # Always re-import: disk shards are the source of truth during the
            # merge loop, and encode_with_vocab rebuilds the right starting
            # state when continuing from an existing vocabulary.
            self._corpus_ids = None
            self._corpus_pattern = None
            encode_with_vocab = bool(self.merges)
            ids = self._import_data(data, encode_with_vocab=encode_with_vocab)
            t1 = time.time()
            print(f'Time spent loading data: {t1-t0:.2f}s')
            with DiskCorpus(ids, self._cpus, work_dir) as corpus:
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
            # Re-import: encode with the current vocab when continuing from an
            # earlier train()/load() (new split wave, or no in-memory corpus).
            encode_with_vocab = bool(self.merges)
            ids = self._import_data(data, encode_with_vocab=encode_with_vocab)
            self._corpus_ids = ids
            self._corpus_pattern = self.pattern
        t1 = time.time()
        if not same_wave:
            print(f'Time spent loading data: {t1-t0:.2f}s')

        with RamCorpus(ids, self._cpus) as corpus:
            self._build_merges(corpus, vocab_size, cap_divisor, max_batch_size, t1, verbose)

    def _build_merges(self, corpus: Corpus, vocab_size: int, cap_divisor: int,
                      max_batch_size: int, t1: float, verbose: bool) -> None:
        """
        Run the batched BPE merge loop against any Corpus backend. This is the
        backend-agnostic core: it only asks `corpus` for the initial pair counts
        and for a merge-batch-then-recount each wave; where the chunk data lives
        and how the work is parallelized is the backend's concern.
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
            if not merges_remaining:   # no need to merge last batch
                break
            stats = corpus.merge_and_recount(pairs_to_merge, mult)
            seen_first.clear()
            seen_last.clear()
            pairs_to_merge.clear()

            if verbose:
                t2 = time.time()
                print(f"Batch {batch_count} merged {num_pairs_to_merge} pairs in {t2-t1:.2f} sec. Merges remaining: {merges_remaining}")
                t1 = t2
