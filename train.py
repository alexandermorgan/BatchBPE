"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BatchTokenizer, MultiplierTokenizer
from datasets import load_dataset
from joblib import Parallel, delayed
import pdb



# open some text and train a vocab of 512 tokens
path = "tests/taylorswift.txt"
text = open(path, "r", encoding="utf-8").read()

# large dataset
path = "/Users/amor/.cache/huggingface/datasets/downloads/0a399b889b781e9d7a9e40349cd800f73a3fc62182e06a084bcd1bb607d8d7ac"
# print(f"Loaded dataset of size {txt.nbytes/1024/1024:.2f} MB") # 3:15:35 start of merge 9 -> 3:19:32 end of merge 9

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# save trained Tokenizers for optional inspection later
tokenizers = {}
timings = {}

for TokenizerClass, name in zip([MultiplierTokenizer], ["multi"]):
    # construct the Tokenizer object and kick off verbose training
    t0 = time.time()
    tokenizer = TokenizerClass(store_dict=True)
    # kwargs = {'path': path, }
    # kwargs = {'path': 'parquet', 'data_files': {'train': path}, 'split': 'train'}
    # p0 = load_dataset(**kwargs)
    # p1 = p0.data['text']
    # print(f'type(p1): {type(p1)}')
    # kwargs = {'path': 'melikocki/preprocessed_shakespeare', 'split': 'train'}
    # kwargs = {'path': 'alpindale/light-novels', 'split': 'train', 'streaming': True}
    # s1 = load_dataset(**kwargs)
    # kwargs = {'path': 'HuggingFaceFW/fineweb-edu', 'name': 'CC-MAIN-2024-10', 'split': 'train', 'streaming': True}
    # kwargs = {'path': 'HuggingFaceFW/fineweb-edu', 'name': 'sample-10BT', 'split': 'train', 'streaming': True}
    # fw = load_dataset(**kwargs)
    # first2 = fw.take(2)
    # t2 = time.time()
    # kwargs = {'path': 'parquet', 'data_files': {'train': path}, 'split': 'train', 'streaming': True}
    # p2 = load_dataset(**kwargs)
    # t3 = time.time()
    t1 = time.time()
    # print(f"Loaded datasets in {t1-t0:.2f} and {t2-t1:.2f} and {t3-t2:.2f} seconds")
    p1 = '/Users/amor/Desktop/Code/AI/tokenization/minbpe/2024-07-22-16:36-dataset-dict.json'
    tokenizer.train(p1, 260, verbose=True)
    pdb.set_trace()

    # writes three files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    # tokenizer.save(prefix)
    # tokenizer.load(prefix + ".model")
    tokenizers[name] = tokenizer

    def process_batch(batch):
        for item in batch:
            tokenizer.encode(item.as_py())
        return 1

    t2 = time.time()
    # ds = load_dataset('parquet', data_files={'train': path}, split='train')
    # batch_size = 345250 // 8 + 1  # Adjust based on your memory and performance needs
    # batches = [ds.data['text'][i:i + batch_size] for i in range(0, 345250, batch_size)]
    # results = Parallel(n_jobs=8)(delayed(process_batch)(batch) for batch in batches)  # Adjust n_jobs as needed
    t3 = time.time()
    test = tokenizer.encode(text)
    res = tokenizer.decode(test)
    t4 = time.time()
    # import pdb; pdb.set_trace()
        # assert(text == res)

    # timings
    timings[name] = [t1-t0, t3-t2, t4-t3]

for name, times in timings.items():
    print('\n*****************************')
    print(f"Training {name} tokenizer took:   {times[0]:.2f} seconds")
    print(f"Encoding took:                   {times[1]:.4f} seconds")
    print(f"Decoding took:                   {times[2]:.4f} seconds")

# uncomment the next line to enter interpreter mode with all the above variables in scope
# import code; code.interact(local=locals())