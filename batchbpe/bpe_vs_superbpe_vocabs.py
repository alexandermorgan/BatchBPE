"""This script builds a gpt2-sized vocab in 2 ways, one 100% BPE the
second 90% BPE and the last 10% open-field SuperBPE merges. The text
used for training was the full FineWeb-Edu 10B token subset (50-60 GB
of text). The first 90% of both vocabs is exactly the same, only the
last 10% was built differently. This is the code I actually ran, but
I first ran the two BPE phases, then exited, then ran the SuperBPE
phase after loading the BPE phase's 90% of merges, then applying them
to the full dataset again without applying any regex split.

The final target vocab size is 50304 but 45298 is 90% of the merges
when accounting for the fact that this tokenizer overwrites the 13
stock dead bytes in utf-8."""

from datasets import load_dataset

from batchbpe import BatchTokenizer


ninety_percent_size = 45298
final_size = 50304

# data = '../fineweb_edu_10B.csv'
# tok = BatchTokenizer()
# tok.train(data, vocab_size=ninety_percent_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64")
# tok.train(data, vocab_size=final_size, verbose=True)
# tok.save(f"../models/FWEdu10B_bpe_size_{final_size}_freq_cutoff_64")



finewebedu_10b_stream = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)
superbpe = BatchTokenizer(dedup=False)
superbpe.load(f"../models/FWEdu10B_bpe_size_{ninety_percent_size}_freq_cutoff_64.model")
superbpe.set_pattern(None)  # override the BPE pattern loaded from the model
superbpe.train(finewebedu_10b_stream, vocab_size=final_size, backend="disk", verbose=True)
superbpe.save(f"../models/FWEdu10B_superbpe_size_{final_size}_freq_cutoff_64")