"""This script builds a gpt2-sized vocab in 2 ways, one 100% BPE the
second 90% BPE and the last 10% open-field SuperBPE merges. The text
used for training was the full FineWeb-Edu 10B token subset (50-60 GB
of text). The first 90% of both vocabs is exactly the same, only the
last 10% was built differently. This is the code I actually ran, but
I first ran the two BPE phases, then exited, then ran the SuperBPE
phase after loading the BPE phase's 90% of merges. Tiktoken applies
that vocabulary with the GPT-4 pretokenization pattern, then BatchBPE
flattens those pieces into one corpus row per document so the final
merge phase can cross the former GPT-4 boundaries.

The final target vocab size is 50304 but 45298 is 90% of the merges
when accounting for the fact that this tokenizer overwrites the 13
stock dead bytes in utf-8."""

from pathlib import Path

from datasets import load_dataset

from batchbpe import BatchTokenizer
from batchbpe.base import GPT4_SPLIT_PATTERN


repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / "models"
# Fresh SuperBPE-from-BPE-90% import. Do not reuse superbpe_corpus_checkpoint:
# that directory was merged in place by the interrupted SuperBPE run.
corpus_checkpoint = repo_root / "superbpe_from_bpe90_checkpoint"

ninety_percent_size = 45298
final_size = 50304

# data = '../fineweb_edu_10B.csv'
# tok = BatchTokenizer()
# tok.train(data, vocab_size=ninety_percent_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64")
# tok.train(data, vocab_size=final_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{final_size}_freq_cutoff_64")



superbpe = BatchTokenizer(dedup=False)
superbpe.load(
    str(models_dir / f"FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64.model")
)
# One corpus row per document lets newly learned SuperBPE merges cross the
# original GPT-4 chunk boundaries. The separate import pattern makes tiktoken
# apply the existing stage-1 vocabulary with fast, normal GPT-4 pretokenization.
superbpe.set_pattern(None)
superbpe.set_import_encoding_pattern(GPT4_SPLIT_PATTERN)

training_data = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

superbpe.train(
    training_data,
    vocab_size=final_size,
    backend="disk",
    work_dir=str(corpus_checkpoint),
    memory_efficient=True,
    records_per_shard=10_000,
    verbose=True)
superbpe.save(str(models_dir / f"FWEdu10B_superbpe_size_{final_size}_freq_cutoff_64"))