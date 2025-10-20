"""
Train our Tokenizers on some data, just to see them in action. Takes <50 sec on my m1 macbook air.
"""
import os
import time
# from batchbpe import QuickTokenizer # SuperTokenizer
from batchbpe import BatchTokenizer #, QuickTokenizer,#SuperTokenizer
# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

dataset = "/Users/amor/Desktop/Code/AI/tokenization/BatchBPE/train-00002-of-00018.parquet"
# kwargs = {'path': 'parquet', 'data_files': {'train': dataset}, 'split': 'train', 'streaming': True}
# p2 = load_dataset(**kwargs)
# ~1GB of text compressed into key-value pairs of str: int. Words appearing fewer than 10 times were filtered out.
p1_data = ['./tests/1GB_of_FineWeb-Edu_10B_sample_freq_cutoff_10.csv']
# # You can also train on a text file. See the README for more details on acceptable file formats.
taylor_swift_text = "tests/taylorswift.txt"   # <- a copy of the wikipedia article on Taylor Swift

pairs = (
    # (SuperTokenizer, 'super'),
    (BatchTokenizer, 'batch'),
    # (QuickTokenizer, 'quick'),
)
# for TokenizerClass, name in zip([SuperTokenizer, BatchTokenizer, QuickTokenizer], ['super', 'batch', 'quick']):
for TokenizerClass, name in pairs:
    t0 = time.time()
    tokenizer = TokenizerClass(store_dict=False)
    tokenizer.train(data=p1_data, vocab_size=50304, verbose=True)   # the more merges you do, the larger the average batch size will be
    # tokenizer.train(data=dataset, vocab_size=50304, T=258, p1_data=p1_data, verbose=True)   # the more merges you do, the larger the average batch size will be
    t1 = time.time()
    # write name.model and name.vocab files in the models directory
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    print(f"Running {name} tokenizer took: {t1-t0:.2f} seconds")
