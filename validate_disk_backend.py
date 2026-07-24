"""
Compare BatchTokenizer train configs (e.g. memory_efficient on vs off).

Loads the first N documents from a parquet file and trains each configured run
to the same vocab_size. Reports approximate peak / average RSS per run and
writes a chart of RSS over normalized progress (0–100% of each run) so runs of
different wall-clock duration share the full x-axis.

Each train runs in its own subprocess so RSS is measured against a clean
process baseline (del/gc alone does not return pages to the OS).

Edit the `runs` list in main() to change which settings are compared.

Usage:
    uv run python validate_disk_backend.py
    uv run python validate_disk_backend.py --parquet 000_00000.parquet --docs 5 --vocab-size 512
    uv run python validate_disk_backend.py --pattern '[^\\n]+'
    uv run python validate_disk_backend.py --pattern none
    uv run python validate_disk_backend.py --pattern none --dedup
    uv run python validate_disk_backend.py --pattern none --docs 800
    uv run python validate_disk_backend.py --plot memory_compare.png
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

import psutil
from pyarrow import parquet

from batchbpe import BatchTokenizer
from batchbpe.base import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN


@dataclass(frozen=True)
class TrainRun:
    """One tokenizer training configuration to compare."""
    name: str
    backend: str = "disk"
    memory_efficient: bool = False


@dataclass
class MemSeries:
    """RSS samples collected inside an isolated train subprocess."""
    peak: int
    average: float
    samples: list[tuple[float, int]]  # (elapsed_s, rss_bytes)


@dataclass
class RunReport:
    """Lightweight leftovers from a finished train run (no tokenizer)."""
    run: TrainRun
    elapsed: float
    done_line: str
    rss_line: str
    mem: MemSeries


def series_label(report: RunReport) -> str:
    """Legend label: ``disk_ME=True_45`` (backend, memory_efficient, whole seconds)."""
    return f"{report.run.backend}_ME={report.run.memory_efficient}_{round(report.elapsed)}s"


def resolve_pattern(value: str | None) -> str | None:
    """Map CLI --pattern to a BatchTokenizer pattern (None = open-field)."""
    if value is None:
        return GPT4_SPLIT_PATTERN
    key = value.strip().lower()
    if key in ("none", "null", "open", "open-field"):
        return None
    if key == "gpt2":
        return GPT2_SPLIT_PATTERN
    if key == "gpt4":
        return GPT4_SPLIT_PATTERN
    return value  # treat as a literal regex


def fmt_bytes(n: float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


class RssMonitor:
    """Lightweight process RSS sampler: peak + average + time series."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak = 0
        self._total = 0
        self._n = 0
        self._t0 = 0.0
        # (elapsed_seconds, rss_bytes) for each sample
        self.samples: list[tuple[float, int]] = []

    def _sample(self) -> None:
        try:
            rss = self._proc.memory_info().rss
        except psutil.Error:
            return
        if rss > self.peak:
            self.peak = rss
        self._total += rss
        self._n += 1
        self.samples.append((time.perf_counter() - self._t0, rss))

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> "RssMonitor":
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()  # final sample after the workload

    @property
    def average(self) -> float:
        return self._total / self._n if self._n else 0.0


def plot_memory_chart(reports: list[RunReport], path: str, n_docs: int) -> None:
    """Plot RSS vs normalized progress (0–100% of each run) for all runs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for report in reports:
        mem = report.mem
        if not mem.samples:
            continue
        times = [t for t, _ in mem.samples]
        rss_mb = [r / (1024 ** 2) for _, r in mem.samples]
        t_max = times[-1] if times[-1] > 0 else 1.0
        progress = [100.0 * t / t_max for t in times]
        ax.plot(progress, rss_mb, label=series_label(report), linewidth=1.5)

    ax.set_xlabel("Progress (% of run)")
    ax.set_ylabel("RSS (MB)")
    ax.set_title(f"Memory consumption over training ({n_docs} docs)")
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote memory chart: {path}")


def load_docs(path: str, n: int) -> list[str]:
    pf = parquet.ParquetFile(path)
    if pf.metadata.num_rows < n:
        sys.exit(f"Error: parquet has {pf.metadata.num_rows} rows, need at least {n}")
    batch = next(pf.iter_batches(columns=["text"], batch_size=n))
    docs = [x.as_py() for x in batch.column("text")]
    if len(docs) < n:
        sys.exit(f"Error: only got {len(docs)} docs from first batch, need {n}")
    return docs[:n]


def train(docs: list[str], vocab_size: int, backend: str, pattern: str | None,
          dedup: bool | None, verbose: bool,
          memory_efficient: bool = False) -> tuple[BatchTokenizer, float, RssMonitor]:
    tok = BatchTokenizer(pattern=pattern, multiprocess=True, dedup=dedup)
    with RssMonitor() as mem:
        t0 = time.perf_counter()
        if backend == "disk":
            with tempfile.TemporaryDirectory(prefix="batchbpe_validate_") as work_dir:
                tok.train(docs, vocab_size, backend="disk", work_dir=work_dir,
                          verbose=verbose, memory_efficient=memory_efficient)
        else:
            tok.train(docs, vocab_size, backend="ram", verbose=verbose,
                      memory_efficient=memory_efficient)
        elapsed = time.perf_counter() - t0
    return tok, elapsed, mem


def _train_in_child(
    parquet: str,
    docs: int,
    vocab_size: int,
    backend: str,
    pattern: str | None,
    dedup: bool | None,
    verbose: bool,
    memory_efficient: bool,
) -> dict:
    """Child-process entry: train once and return picklable RSS/report stats."""
    docs_list = load_docs(parquet, docs)
    tok, elapsed, mem = train(
        docs_list, vocab_size, backend, pattern, dedup, verbose, memory_efficient)
    return {
        "elapsed": elapsed,
        "n_merges": len(tok.merges),
        "n_vocab": len(tok.vocab),
        "peak": mem.peak,
        "average": mem.average,
        "samples": mem.samples,
    }


def run_isolated(
    run: TrainRun,
    parquet: str,
    docs: int,
    vocab_size: int,
    pattern: str | None,
    dedup: bool | None,
    verbose: bool,
) -> RunReport:
    """Train in a fresh subprocess so RSS starts from a clean process baseline."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.apply(
            _train_in_child,
            (parquet, docs, vocab_size, run.backend, pattern, dedup, verbose,
             run.memory_efficient),
        )
    done_line = (f"{run.name} done in {result['elapsed']:.2f}s  "
                 f"merges={result['n_merges']} vocab={result['n_vocab']}")
    rss_line = (f"{run.name} RSS peak={fmt_bytes(result['peak'])}  "
                f"avg≈{fmt_bytes(result['average'])}")
    mem = MemSeries(peak=result["peak"], average=result["average"],
                    samples=result["samples"])
    return RunReport(run=run, elapsed=result["elapsed"], done_line=done_line,
                     rss_line=rss_line, mem=mem)


def print_run_settings(run: TrainRun, vocab_size: int, pattern: str | None,
                       pattern_label: str, dedup: bool | None) -> None:
    if dedup is None:
        dedup_label = f"auto ({'on' if pattern is not None else 'off'})"
    else:
        dedup_label = "on" if dedup else "off"
    print(f"\n--- {run.name} ---")
    print(f"  backend         : {run.backend}")
    print(f"  memory_efficient: {run.memory_efficient}")
    print(f"  vocab_size      : {vocab_size}")
    print(f"  pattern         : {pattern_label}")
    print(f"  dedup           : {dedup_label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BatchTokenizer train configs for merge/vocab parity.",
    )
    parser.add_argument(
        "--parquet",
        default="000_00000.parquet",
        help="Path to the .parquet file (default: 000_00000.parquet)",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=5,
        help="Number of leading documents to use (default: 5)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Target vocabulary size (default: 512)",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Split pattern: omit for GPT-4 default; 'gpt2' / 'gpt4' aliases; "
             "'none' for open-field (no split); or any regex string "
             "(e.g. '[^\\n]+' for newline-only SuperBPE-style chunks).",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Counter-dedup on/off. Default: auto (on with a split pattern, "
             "off for open-field). Use --dedup / --no-dedup to override.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch merge progress",
    )
    parser.add_argument(
        "--plot",
        default="memory_compare.png",
        help="Path for the RSS-over-progress chart (default: memory_compare.png). "
             "Pass empty string to skip plotting.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.parquet):
        sys.exit(f"Error: parquet file not found: {args.parquet}")
    if args.docs < 1:
        sys.exit("Error: --docs must be >= 1")
    if args.vocab_size <= 256:
        sys.exit("Error: --vocab-size must be > 256 (byte tokens)")

    pattern = resolve_pattern(args.pattern)
    if pattern is None:
        pattern_label = "none (open-field)"
    elif args.pattern is None or args.pattern.strip().lower() == "gpt4":
        pattern_label = "gpt4 (default)"
    elif args.pattern.strip().lower() == "gpt2":
        pattern_label = "gpt2"
    else:
        pattern_label = repr(pattern)

    ############################################################################
    # Edit this list to add/change comparison runs. Shared CLI knobs (parquet,
    # docs, vocab, pattern, dedup, verbose) still apply to every run.
    runs = [
        TrainRun(name="memory_efficient", backend="disk", memory_efficient=True),
        # TrainRun(name="defaults", backend="ram", memory_efficient=False),
    ]
    ############################################################################

    docs = load_docs(args.parquet, args.docs)
    total_chars = sum(len(d) for d in docs)
    print(f"Parquet file : {args.parquet}")
    print(f"Documents    : {len(docs)}  ({total_chars:,} chars)")
    print(f"Comparing    : {', '.join(r.name for r in runs)}")
    del docs

    reports: list[RunReport] = []
    for run in runs:
        print_run_settings(run, args.vocab_size, pattern, pattern_label, args.dedup)
        report = run_isolated(
            run, args.parquet, args.docs, args.vocab_size, pattern, args.dedup,
            args.verbose)
        print(report.done_line)
        print(report.rss_line)
        reports.append(report)

    # Parity comparison disabled: each run is isolated in a subprocess and only
    # returns RSS/report stats (no tokenizer objects to compare).
    # (run_a, tok_a, _, _), (run_b, tok_b, _, _) = results[0], results[1]
    # sample = "\n".join(docs)[:4000]
    # merges_ok = tok_a.merges == tok_b.merges
    # vocab_ok = tok_a.vocab == tok_b.vocab
    # encode_ok = tok_a.encode(sample) == tok_b.encode(sample)
    # print(f"\n=== PARITY ({run_a.name} vs {run_b.name}) ===")
    # print(f"merges identical:        {merges_ok}")
    # print(f"vocab identical:         {vocab_ok}")
    # print(f"encode sample identical: {encode_ok}")

    print("\n=== MEMORY ===")
    for report in reports:
        print(f"{report.run.name:20s} peak={fmt_bytes(report.mem.peak)}  "
              f"avg≈{fmt_bytes(report.mem.average)}")

    if args.plot:
        plot_memory_chart(reports, args.plot, args.docs)

    # if merges_ok and vocab_ok:
    #     print(f"\nPASS: {run_b.name} matches {run_a.name} on first "
    #           f"{args.docs} docs of {args.parquet}")
    #     return
    # if not merges_ok:
    #     a_only = set(tok_a.merges) - set(tok_b.merges)
    #     b_only = set(tok_b.merges) - set(tok_a.merges)
    #     print(f"  {run_a.name}-only pairs: {len(a_only)}  "
    #           f"{run_b.name}-only pairs: {len(b_only)}")
    #     for k, v in tok_a.merges.items():
    #         if tok_b.merges.get(k) != v:
    #             print(f"  first diverge: {k} {run_a.name}={v} "
    #                   f"{run_b.name}={tok_b.merges.get(k)}")
    #             break
    # sys.exit(1)

if __name__ == "__main__":
    main()
