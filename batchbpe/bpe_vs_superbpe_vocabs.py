"""Two-stage FineWeb-Edu SuperBPE experiment.

Stage 1 (already done): ordinary GPT-4-split BPE to 45298, saved as
``models/FWEdu10B_bpe_size_45298_freq_cutoff_64.model``.

Stage 2 (this script):
  1. Stream HuggingFaceFW/fineweb-edu sample-10BT.
  2. Encode each document with the loaded 45298 model via tiktoken
     (GPT-4 pretokenization for import only), writing one open-field
     corpus row per document into a fresh DiskCorpus.
  3. Continue merges with Rust disk backend from that 45298 vocabulary
     up to 50304 under ``pattern=None`` (true SuperBPE / open-field).
  4. Save ``FWEdu10B_superbpe_size_50304_freq_cutoff_64_<date>.{model,vocab}``.

Why prior SuperBPE artifacts are not reusable as this experiment's output:
  - SuperBPE must *continue* the 45298 merges, not train a fresh 50304 vocab.
  - Disk shards are merged in place. After a SuperBPE run they contain token
    IDs from the full 50304 vocab, so ``resume_from_manifest`` on that corpus
    while reloading only the 45298 model trains on the wrong token sequences.
  - Do not reuse ``superbpe_bpe90_corpus_*`` dirs left over from earlier runs;
    this script always writes a new dated corpus directory.

Peak RAM while encoding: one shard buffer (``records_per_shard``) plus a
small tiktoken batch — never the full FineWeb stream.
"""

from datetime import date
from pathlib import Path

from datasets import load_dataset

from batchbpe import BatchTokenizer
from batchbpe.base import GPT4_SPLIT_PATTERN
from batchbpe.corpus import DiskCorpus


repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / "models"

ninety_percent_size = 45298
final_size = 50304
run_date = date.today().isoformat()

# Fresh corpus every run. Never point this at an already SuperBPE-merged dir.
corpus_checkpoint = repo_root / f"superbpe_bpe90_corpus_{run_date}"
stage1_model = models_dir / f"FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64.model"
out_prefix = models_dir / f"FWEdu10B_superbpe_size_{final_size}_freq_cutoff_64_{run_date}"

# data = '../fineweb_edu_10B.csv'
# tok = BatchTokenizer()
# tok.train(data, vocab_size=ninety_percent_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64")
# tok.train(data, vocab_size=final_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{final_size}_freq_cutoff_64")


# ---------------------------------------------------------------------------
# Load stage-1 BPE (45298). All SuperBPE merges are added on top of this.
# ---------------------------------------------------------------------------
superbpe = BatchTokenizer(dedup=False)
superbpe.load(str(stage1_model))
assert len(superbpe.vocab) == ninety_percent_size, (
    f"expected stage-1 vocab size {ninety_percent_size}, got {len(superbpe.vocab)}"
)

# Open-field merge boundaries (SuperBPE). Import still uses GPT-4 pretok so
# tiktoken applies the existing 45298 vocab the same way stage 1 did, then
# flattens each document into one corpus row.
superbpe.set_pattern(None)
superbpe.set_import_encoding_pattern(GPT4_SPLIT_PATTERN)

# ---------------------------------------------------------------------------
# Step 1: stream FineWeb-Edu → fresh disk shards at the 45298 vocabulary.
# ---------------------------------------------------------------------------
if corpus_checkpoint.exists():
    raise SystemExit(
        f"Refusing to reuse existing corpus dir {corpus_checkpoint}. "
        "Pick a new run_date or remove that directory only if you are sure "
        "it is an unmerged 45298 encode (not a prior SuperBPE merge)."
    )

training_data = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

print(f"Encoding FineWeb-Edu sample-10BT with {stage1_model.name} → {corpus_checkpoint}")
corpus_checkpoint.mkdir(parents=True, exist_ok=False)
DiskCorpus.from_chunk_iter(
    superbpe._iter_chunk_arrays(training_data, encode_with_vocab=True),
    superbpe._cpus,
    str(corpus_checkpoint),
    records_per_shard=10_000,
).close()
print(f"Finished encoding into {corpus_checkpoint}")

# ---------------------------------------------------------------------------
# Step 2: Rust disk SuperBPE continues from the loaded 45298 merges → 50304.
# ---------------------------------------------------------------------------
print(
    f"Continuing SuperBPE open-field merges from {ninety_percent_size} → {final_size} "
    f"on {corpus_checkpoint}"
)
superbpe.train(
    None,
    vocab_size=final_size,
    backend="disk",
    resume_from_manifest=str(corpus_checkpoint),
    memory_efficient=True,
    verbose=True,
)

assert len(superbpe.vocab) == final_size, (
    f"expected final vocab size {final_size}, got {len(superbpe.vocab)}"
)
superbpe.save(str(out_prefix))
print(f"Saved {out_prefix}.model and {out_prefix}.vocab")
