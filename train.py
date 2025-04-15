"""
Train our Tokenizers on some data, just to see them in action. Takes <50 sec on my m1 macbook air.
"""
import os
import time
from batchbpe import BatchTokenizer, QuickTokenizer
# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# ~1GB of text compressed into key-value pairs of str: int. Words appearing fewer than 10 times were filtered out.
data = ['./tests/1GB_of_FineWeb-Edu_10B_sample_freq_cutoff_10.csv']
# # You can also train on a text file. See the README for more details on acceptable file formats.
taylor_swift_text = "tests/taylorswift.txt"   # <- a copy of the wikipedia article on Taylor Swift

for TokenizerClass, name in zip([BatchTokenizer, QuickTokenizer], ['batch', 'quick']):
    t0 = time.time()
    tokenizer = TokenizerClass(store_dict=False)
    tokenizer.train(data, 50304, verbose=True)   # the more merges you do, the larger the average batch size will be
    t1 = time.time()
    # write name.model and name.vocab files in the models directory
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    print(f"Running {name} tokenizer took: {t1-t0:.2f} seconds")
