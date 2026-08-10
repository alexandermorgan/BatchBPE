"""
Corpus backends for the batched BPE merge loop.

The merge loop (BatchTokenizer._build_merges) is storage-agnostic: it only asks
a Corpus for an initial pair count and for a "merge this batch of pairs in place,
then recount" step. Concrete backends decide where the chunk data lives and how
the work is parallelized:

  - RamCorpus:  the chunk list lives in memory (the typical case).
  - DiskCorpus: sharded on-disk backend for SuperBPE-style continued training on
    datasets too large for RAM and/or open-field merges (e.g. not space-split).
    Chunks live in fixed-capacity slots inside small shard files; only shortened
    docs are rewritten in place. A bounded worker pool keeps shards in flight
    with work-stealing; count-dicts are still tree-combined in worker-sized batches.

Both backends accept `max_stats_size` (from train(memory_efficient=True)): when
positive, pair-count dicts are kept to that many keys by repeatedly retaining
the true top counts (exact values for survivors). Truncation runs per shard,
after each combine step, and on the running disk aggregate so peak stats RAM
stays O(max_stats_size) rather than O(unique pairs in the corpus). Early
truncation can slightly change merge order vs the unbounded path.

The chunk record format is shared with get_stats / merge_batch_and_get_stats:
each chunk is `array('i', [count, *token_ids])`. Pairs (and the returned counts)
are keyed by the packed int `first*mult + last`, which avoids allocating a tuple
per adjacent pair in the hot loop; `mult` must exceed the largest token id so
that divmod recovers (first, last).

Docs are handled one at a time: merge (if any) then count for that chunk before
moving on. Disk shards use fixed-capacity slots so a shortened doc is rewritten
in place without touching its neighbors; RamCorpus still retains its list for
the whole run but uses the same per-doc control flow.
"""
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from heapq import nlargest
from itertools import batched
import json
import os
import shutil
import tempfile


def _shards(seq: list[array[int]], n: int) -> batched[tuple[array[int], ...]]:
    """Split `seq` into at most `n` contiguous batches for parallel workers."""
    size = (len(seq) + n - 1) // n or 1
    return batched(seq, size)


def _bound_stats(counts: defaultdict[int, int], capacity: int) -> defaultdict[int, int]:
    """Keep at most `capacity` keys with exact counts (true top-k by count, then key)."""
    if capacity <= 0 or len(counts) <= capacity:
        return counts
    return defaultdict[int, int](
        int, nlargest(capacity, counts.items(), key=lambda kv: (kv[1], kv[0])))


def _merge_two(a: defaultdict[int, int], b: defaultdict[int, int]) -> defaultdict[int, int]:
    """Sum count dict `b` into `a` (the larger of the two) and return it."""
    if len(b) > len(a):
        a, b = b, a
    for k, v in b.items():
        a[k] += v
    return a


def _combine_counts(parts: list[defaultdict[int, int]], pool: ThreadPoolExecutor | None,
                    capacity: int = 0) -> defaultdict[int, int]:
    """Sum a list of packed-int count dicts into one (the global pair stats).

    Reduced as a balanced binary tree: each wave merges disjoint pairs of dicts,
    halving the count per wave, so the merge depth is ceil(log2(len(parts)))
    waves (e.g. 8 shards -> 3, 16 -> 4, 32 -> 5) instead of len(parts)-1 serial
    merges. When `pool` is set, each wave runs concurrently on it; when `pool`
    is None, merges run on the caller thread (avoids competing with in-flight
    shard work on the same executor). When `capacity > 0`, each merge result is
    truncated to that many keys so intermediates never grow with corpus size.
    """
    while len(parts) > 1:
        pairs = [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]
        if pool is None:
            merged = [_merge_two(a, b) for a, b in pairs]
        else:
            merged = list[defaultdict[int, int]](pool.map(lambda ab: _merge_two(*ab), pairs))
        if capacity > 0:
            merged = [_bound_stats(m, capacity) for m in merged]
        if len(parts) % 2:   # odd one out carries forward to the next wave
            merged.append(parts[-1])
        parts = merged
    return _bound_stats(parts[0], capacity)


def _merge_chunk(chunk: array[int], pairs_get: Callable[[int], int | None],
                 mult: int) -> None:
    """Apply packed `pairs` to one chunk in place (left-to-right)."""
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


def _accumulate_chunk_stats(chunk: array[int], counts: defaultdict[int, int],
                            mult: int) -> None:
    """Add one chunk's adjacent-pair counts into `counts` (packed keys)."""
    num = chunk[0]
    second_last_index = len(chunk) - 2  # second-to-last token index
    i = 1  # Start at index 1 (skip count)
    while i < second_last_index:
        j = i + 1
        counts[chunk[i] * mult + chunk[j]] += num
        if chunk[i] == chunk[j] == chunk[i + 2]:
            i += 2  # skip next token to avoid overcounting consecutive repeated pairs
        else:
            i = j
    if i == second_last_index:
        counts[chunk[i] * mult + chunk[i + 1]] += num


def get_stats(ids: Iterable[array[int]], mult: int,
              max_stats_size: int = 0) -> defaultdict[int, int]:
    """
    Given `ids`, an iterable of chunks where each chunk contains a count as the
    FIRST element followed by tokens, returns a defaultdict with the
    counts of occurrences of all consecutive pairs of integers within each
    list, multiplied by the count value. Consecutive identical pairs within
    the same list are counted only once to avoid overcounting repeat characters.

    Chunks are consumed one at a time (friendly to generators / disk streams).

    Pairs are keyed by the packed int `first*mult + last` instead of a
    `(first, last)` tuple, which avoids allocating a throwaway tuple for every
    adjacent pair in the hot loop. `mult` must exceed the largest token id
    (vocab_size) so that divmod recovers (first, last).

    When `max_stats_size > 0`, counts are pruned to that many keys whenever the
    dict grows past 2x the cap (and again at the end), so peak size stays
    O(max_stats_size). Survivors keep exact counts; pruned keys are dropped.

    Example (mult=1000):
        get_stats([[2, 97, 98, 99], [1, 98, 99, 100], [1, 101, 101, 101]], 1000)
        -> defaultdict(<class 'int'>, {97098: 2, 98099: 3, 99100: 1, 101101: 1})
    """
    counts = defaultdict[int, int](int)
    soft_limit = max_stats_size * 2 if max_stats_size > 0 else 0
    for chunk in ids:
        _accumulate_chunk_stats(chunk, counts, mult)
        if soft_limit and len(counts) > soft_limit:
            counts = _bound_stats(counts, max_stats_size)
    return _bound_stats(counts, max_stats_size)


def merge_batch_and_get_stats(ids: Iterable[array[int]], pairs: dict[int, int],
                              mult: int,
                              max_stats_size: int = 0) -> defaultdict[int, int]:
    """
    For each chunk: merge `pairs` in place, then add that chunk's pair counts,
    then move on. Both `pairs` and the returned counts use packed keys
    `first*mult + last`. Merge and recount stay separate tight loops per doc
    (not one mixed token-level walk).
    """
    counts = defaultdict[int, int](int)
    soft_limit = max_stats_size * 2 if max_stats_size > 0 else 0
    pairs_get = pairs.get
    for chunk in ids:
        _merge_chunk(chunk, pairs_get, mult)
        _accumulate_chunk_stats(chunk, counts, mult)
        if soft_limit and len(counts) > soft_limit:
            counts = _bound_stats(counts, max_stats_size)
    return _bound_stats(counts, max_stats_size)


class Corpus(ABC):
    """Backend-agnostic view of the training corpus used by the merge loop.

    The batched merge loop only needs two things from the corpus: an initial
    pair count over every chunk, and a "merge this batch of pairs in place, then
    recount" step. Concrete backends decide where the chunk data actually lives
    (in RAM, on disk, ...) and how the work is parallelized. Pairs and the
    returned counts are keyed by the packed int `first*mult + last`.
    """
    @abstractmethod
    def initial_stats(self, mult: int) -> defaultdict[int, int]:
        """Return the pair counts over every chunk in the corpus."""

    @abstractmethod
    def merge_and_recount(self, pairs_to_merge: dict[int, int], mult: int) -> defaultdict[int, int]:
        """Apply `pairs_to_merge` to every chunk in place, then return new counts."""

    def close(self) -> None:
        pass

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RamCorpus(Corpus):
    """In-memory backend (the typical case): the deduplicated chunk list lives
    in RAM. Chunks are sharded once and merged in place across batches, and a
    single thread pool is reused for the whole run. Within each shard worker,
    docs are merged then counted one at a time (same control flow as DiskCorpus).

    When `max_stats_size > 0`, pair-count dicts are truncated to that many keys
    throughout recount/combine (see _bound_stats).
    """
    def __init__(self, ids: list[array[int]], n: int, max_stats_size: int = 0) -> None:
        self._shards = [*_shards(ids, n)]
        self._pool = ThreadPoolExecutor(max_workers=n)
        self._max_stats_size = max(0, max_stats_size)

    def initial_stats(self, mult: int) -> defaultdict[int, int]:
        cap = self._max_stats_size
        return _combine_counts(list(self._pool.map(
            lambda shard: get_stats(shard, mult, cap), self._shards)),
            self._pool, cap)

    def merge_and_recount(self, pairs_to_merge: dict[int, int], mult: int) -> defaultdict[int, int]:
        cap = self._max_stats_size
        return _combine_counts(list(self._pool.map(
            lambda shard: merge_batch_and_get_stats(shard, pairs_to_merge, mult, cap),
            self._shards)), self._pool, cap)

    def close(self) -> None:
        self._pool.shutdown()


# -----------------------------------------------------------------------------
# on-disk shard I/O — fixed-capacity slots (shrink-only in-place updates)
#
# Each slot is:
#   uint32 capacity   # max int32 payload elements (fixed at first write)
#   uint32 used       # current int32 payload elements (shrinks on merge)
#   int32[capacity]   # payload; only the first `used` elements are live
#
# Slot byte size is fixed at create time, so a shortened chunk is rewritten in
# place without shifting later docs. Slack (capacity - used) is left as padding.

_MANIFEST_NAME = "manifest.json"
_SHARD_FMT = "shard_{:06d}.bin"
_SCHEMA_VERSION = 3  # v3: fixed-capacity slots; in-place shrink on merge
_RECORDS_PER_SHARD = 100  # dataset records (chunks) per on-disk shard file


def _append_chunk(f, chunk: array[int]) -> None:
    """Append one fixed-capacity slot (capacity == used at create)."""
    n = len(chunk)
    f.write(array("I", [n, n]).tobytes() + chunk.tobytes())


def _write_shard(path: str, chunks: list[array[int]] | tuple[array[int], ...]) -> None:
    """Write chunks as a stream of fixed-capacity slots."""
    with open(path, "wb") as f:
        for chunk in chunks:
            _append_chunk(f, chunk)


def _read_slot_at(f) -> tuple[int, int, array[int]] | None:
    """Read the next slot. Returns (offset, capacity, chunk) or None at EOF.

    Leaves the file position at the start of the following slot. `chunk` holds
    only the live `used` elements (slack is skipped).
    """
    offset = f.tell()
    hdr = array("I")
    try:
        hdr.fromfile(f, 2)
    except EOFError:
        return None
    capacity, used = int(hdr[0]), int(hdr[1])
    if used > capacity:
        raise ValueError(f"corrupt slot at {offset}: used={used} > capacity={capacity}")
    chunk = array("i")
    chunk.fromfile(f, used)
    if capacity > used:
        f.seek((capacity - used) * 4, os.SEEK_CUR)
    return offset, capacity, chunk


def _write_slot_inplace(f, offset: int, capacity: int, chunk: array[int],
                        resume_at: int) -> None:
    """Overwrite a slot's used-count + payload; leave slack and later slots alone."""
    used = len(chunk)
    if used > capacity:
        raise ValueError(f"chunk len {used} exceeds slot capacity {capacity}")
    f.seek(offset)
    f.write(array("I", [capacity, used]).tobytes() + chunk.tobytes())
    f.seek(resume_at)


def _iter_chunks(f) -> Iterator[array[int]]:
    """Yield live chunk payloads from slotted shard file until EOF."""
    while True:
        slot = _read_slot_at(f)
        if slot is None:
            break
        yield slot[2]


def _iter_shard(path: str) -> Iterator[array[int]]:
    """Stream chunks from a shard file one at a time."""
    with open(path, "rb") as f:
        yield from _iter_chunks(f)


def _read_shard(path: str) -> list[array[int]]:
    """Read an entire shard into a list (tests / callers that need random access)."""
    return list(_iter_shard(path))


class DiskCorpus(Corpus):
    """Sharded on-disk backend: chunks live in work_dir as slotted binary files.

    Each shard holds `records_per_shard` dataset records (default
    `_RECORDS_PER_SHARD`) in fixed-capacity slots. A worker pool of size `n`
    keeps up to `n` shard tasks in flight (a free worker takes the next shard
    immediately). Completed shard count-dicts are tree-combined in batches of
    `n` — same combine shape as a wave barrier, without idle gaps. Within a
    shard, each doc is merged (if needed) and counted one at a time; only docs
    that shorten are rewritten in place. Peak token RAM is about one chunk per
    worker. Build from an in-memory list, or stream with `from_chunk_iter`
    (no full-corpus RAM list).

    When `max_stats_size > 0`, pair-count dicts are truncated to that many keys
    throughout recount/combine (see _bound_stats).
    """

    def __init__(self, ids: list[array[int]] | None = None, n: int = 1,
                 work_dir: str | None = None, *,
                 records_per_shard: int = _RECORDS_PER_SHARD,
                 max_stats_size: int = 0,
                 _shard_paths: list[str] | None = None,
                 _owns_dir: bool | None = None) -> None:
        if _owns_dir is None:
            self._owns_dir = work_dir is None
        else:
            self._owns_dir = _owns_dir
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="batchbpe_corpus_")
        else:
            os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self._n_workers = max(1, n)
        self._records_per_shard = max(1, records_per_shard)
        self._max_stats_size = max(0, max_stats_size)
        if _shard_paths is not None:
            self._shard_paths = _shard_paths
        else:
            if ids is None:
                raise ValueError("DiskCorpus requires ids=... or from_chunk_iter(...)")
            self._shard_paths = self._materialize(ids)
        self._pool = ThreadPoolExecutor(max_workers=self._n_workers)

    @classmethod
    def from_chunk_iter(cls, chunks: Iterable[array[int]], n: int,
                        work_dir: str | None = None,
                        records_per_shard: int = _RECORDS_PER_SHARD,
                        max_stats_size: int = 0) -> "DiskCorpus":
        """Stream `chunks` into small shard files without buffering the full list."""
        owns_dir = work_dir is None
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="batchbpe_corpus_")
        else:
            os.makedirs(work_dir, exist_ok=True)
        records_per_shard = max(1, records_per_shard)
        paths: list[str] = []
        chunk_counts: list[int] = []
        buf: list[array[int]] = []
        shard_i = 0

        def _flush() -> None:
            nonlocal shard_i, buf
            if not buf:
                return
            path = os.path.join(work_dir, _SHARD_FMT.format(shard_i))
            _write_shard(path, buf)
            paths.append(path)
            chunk_counts.append(len(buf))
            shard_i += 1
            buf = []

        for chunk in chunks:
            buf.append(chunk)
            if len(buf) >= records_per_shard:
                _flush()
        _flush()

        manifest = {
            "version": _SCHEMA_VERSION,
            "n_shards": len(paths),
            "records_per_shard": records_per_shard,
            "shards": [os.path.basename(p) for p in paths],
            "chunk_counts": chunk_counts,
        }
        with open(os.path.join(work_dir, _MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        return cls(n=n, work_dir=work_dir, records_per_shard=records_per_shard,
                   max_stats_size=max_stats_size,
                   _shard_paths=paths, _owns_dir=owns_dir)

    def _materialize(self, ids: list[array[int]]) -> list[str]:
        """Write ids as contiguous shards of `records_per_shard` chunks each."""
        paths: list[str] = []
        chunk_counts: list[int] = []
        for i, shard in enumerate(batched(ids, self._records_per_shard)):
            path = os.path.join(self.work_dir, _SHARD_FMT.format(i))
            _write_shard(path, shard)
            paths.append(path)
            chunk_counts.append(len(shard))
        manifest = {
            "version": _SCHEMA_VERSION,
            "n_shards": len(paths),
            "records_per_shard": self._records_per_shard,
            "shards": [os.path.basename(p) for p in paths],
            "chunk_counts": chunk_counts,
        }
        with open(os.path.join(self.work_dir, _MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        return paths

    def _stats_from_path(self, path: str, mult: int) -> defaultdict[int, int]:
        return get_stats(_iter_shard(path), mult, self._max_stats_size)

    def _merge_from_path(self, path: str, pairs: dict[int, int],
                         mult: int) -> defaultdict[int, int]:
        """Stream one doc at a time: merge and count; rewrite only shortened slots.

        Merges only shorten chunks. Unchanged docs are left untouched. Shortened
        docs are written back into their fixed-capacity slot in place so later
        docs keep their byte offsets (no shard-wide tmp rewrite).
        """
        cap = self._max_stats_size
        soft_limit = cap * 2 if cap > 0 else 0
        counts = defaultdict[int, int](int)
        pairs_get = pairs.get
        with open(path, "r+b") as f:
            while True:
                slot = _read_slot_at(f)
                if slot is None:
                    break
                offset, capacity, chunk = slot
                resume_at = f.tell()
                n_before = len(chunk)
                _merge_chunk(chunk, pairs_get, mult)
                _accumulate_chunk_stats(chunk, counts, mult)
                if soft_limit and len(counts) > soft_limit:
                    counts = _bound_stats(counts, cap)
                if len(chunk) != n_before:
                    _write_slot_inplace(f, offset, capacity, chunk, resume_at)
        return _bound_stats(counts, cap)

    def _reduce_paths(self, fn) -> defaultdict[int, int]:
        """Run `fn` on every shard with work-stealing; combine in waves of `n`.

        Up to `n_workers` shard tasks stay in flight: when one finishes, the
        next pending shard is submitted immediately (no barrier). Completed
        count-dicts are buffered and tree-combined in batches of `n_workers`
        via `_combine_counts` — same combine cardinality as the old wave
        scheduler — then folded into a running total. Batch combines run on
        the caller thread so they do not contend with in-flight shard tasks on
        the pool. When `max_stats_size > 0`, each combine/fold truncates so
        peak stats RAM stays O(max_stats_size).
        """
        paths = self._shard_paths
        if not paths:
            return defaultdict[int, int](int)
        n = self._n_workers
        cap = self._max_stats_size
        path_iter = iter(paths)
        in_flight: set = set()

        def _fill() -> None:
            while len(in_flight) < n:
                try:
                    path = next(path_iter)
                except StopIteration:
                    break
                in_flight.add(self._pool.submit(fn, path))

        def _fold_batch(batch: list[defaultdict[int, int]],
                        total: defaultdict[int, int] | None) -> defaultdict[int, int]:
            # pool=None: combine on this thread so shard workers keep stealing.
            wave_total = (_combine_counts(batch, None, cap)
                          if len(batch) > 1 else _bound_stats(batch[0], cap))
            return (wave_total if total is None
                    else _bound_stats(_merge_two(total, wave_total), cap))

        _fill()
        buffer: list[defaultdict[int, int]] = []
        total: defaultdict[int, int] | None = None
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                in_flight.remove(fut)
                buffer.append(fut.result())
            # Combine before refill so we release batch dicts promptly.
            while len(buffer) >= n:
                total = _fold_batch(buffer[:n], total)
                del buffer[:n]
            _fill()
        if buffer:
            total = _fold_batch(buffer, total)
        assert total is not None
        return total

    def initial_stats(self, mult: int) -> defaultdict[int, int]:
        return self._reduce_paths(lambda path: self._stats_from_path(path, mult))

    def merge_and_recount(self, pairs_to_merge: dict[int, int], mult: int) -> defaultdict[int, int]:
        return self._reduce_paths(
            lambda path: self._merge_from_path(path, pairs_to_merge, mult))

    def close(self) -> None:
        self._pool.shutdown()
        if self._owns_dir and os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
