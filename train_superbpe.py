"""
SuperBPE-style vocabulary building by calling BatchTokenizer.train() twice.

Stage 1 runs ordinary BPE with the GPT-4 word-level split pattern. Stage 2
reuses the very same tokenizer (its stage-1 vocabulary intact) but swaps in a
liberal "split on newlines only" pattern and keeps merging, so merges are now
allowed to cross the boundaries the GPT-4 pattern used to enforce.

Stage 2 reuses the stage-1 vocabulary but changes the split pattern, so the
corpus is reloaded and each chunk is encoded with the stage-1 merges before
the next merge wave begins.

Total vocab: 1,500 tokens
  - stage 1: up to 1,350 tokens (the ~256 byte tokens + ordinary merges)
  - stage 2: 150 further merges using "\n" as the only split boundary
"""
import os

import regex as re

from batchbpe import BatchTokenizer
from batchbpe.base import GPT4_SPLIT_PATTERN

CORPUS = "tests/taylorswift.txt"
STAGE1_VOCAB_SIZE = 1350   # ~256 byte tokens + ordinary merges
FINAL_VOCAB_SIZE = 1500    # + 150 superword (newline-only) merges
# findall-equivalent of "split on \n": each non-empty line is a single chunk,
# so the only boundary a merge cannot cross is a newline.
NEWLINE_SPLIT_PATTERN = r"[^\n]+"

os.makedirs("models", exist_ok=True)

# --- Stage 1: ordinary BPE up to 1,350 tokens, saved as its own files ---
tok = BatchTokenizer(pattern=GPT4_SPLIT_PATTERN)
tok.train(data=CORPUS, vocab_size=STAGE1_VOCAB_SIZE, verbose=True)
tok.save(os.path.join("models", "stage1"))
print(f"Stage 1 done: {len(tok.vocab)} tokens")

# Stage 2: new split wave — corpus is reloaded and pre-encoded with stage-1 vocab.
tok.pattern = NEWLINE_SPLIT_PATTERN
tok.compiled_pattern = re.compile(NEWLINE_SPLIT_PATTERN)
tok.train(data=CORPUS, vocab_size=FINAL_VOCAB_SIZE, verbose=True)
tok.save(os.path.join("models", "superbpe_taylorswift"))
print(f"Stage 2 done: {len(tok.vocab)} tokens")
