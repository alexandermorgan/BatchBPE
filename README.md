# BatchBPE

Practical, performant, pure python implementation of a Byte Pair Encoding (BPE) tokenizer. The "Batch" part of the name is for the most characteristic aspect of BatchBPE, which is that it executes token-pair merges safely in batches. ~200 is the average batch size when building a ~50k token vocabulary based on English training texts.

Here is the shortest useful demonstration of what BatchBPE can do (after installing): load 1GB's worth of text into a tokenizer and train a ~50k token vocabulary. Don't worry, it doesn't download any data:

```python
from batchbpe import QuickTokenizer
tokenizer = QuickTokenizer()
data = 'path_to_sample_dict_of_1gb_of_text_with_freq_cutoff_10.json'
tokenizer.train(data, 50304, verbose=True)
```

The example above runs in less than a minute on an old laptop, not bad for a pure python implementation! The purpose of this repo is to be easy to use and tinker with. I hope it helps people think up and quickly try out new tokenization approaches.

## Origin Story

This repo is a fork of Andrej Karpathy's excellent introduction to the BPE used for LLM tokenization. If you're new to the subject but haven't reviewed Karpathy's resources, definitely start there. He has a [2-hour video lecture](https://www.youtube.com/watch?v=zduSFxRajkE) (and [text version of lecture](https://github.com/karpathy/minbpe/blob/master/lecture.md)), accompanying [minbpe github repo](https://github.com/karpathy/minbpe), and [colab notebook](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbEtrVFZtbHhpLUtxWE5aeVNIaUlSNkhpWHdVUXxBQ3Jtc0tuWE9pbHBPZmF2anlYeTZfdTlVXzYyTmREeDNEejZMYnctNk96UnFuMjZBTUVHemkyWjdlWEhYSE56LUNsVFJrakNXeng3NEQxREkwLUFlQWpKa1JHd3JfX3k5dU5TVWFoQzNnWU9XY0lPUElUTUtydw&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L%3Fusp%3Dsharing&v=zduSFxRajkE). I highly recommend all three. Tokenization is deceptively simple, so a deep dive into the topic is definitely worth it even if you can understand the basics with a 60-second intro.

This BatchBPE repo began as a PR for Karpath's minbpe but developed to the point where the objective changed. Instead of minbpe's pedagogic purpose, BatchBPE aims to be as practical and easy to modify as possible. The goal is to make it easy for people to try out new tokenization ideas even if they're working with limited compute, memory, or hard-disk resources. A lot of making tokenization more accessible boils down to compute and memory optimizations. Using BatchBPE's fastest combination of settings (described below), you can train a GPT2-sized vocabulary (~50k tokens) on 1GB's worth of text in well under a minute on a 4-year old laptop. So yes, trying out new tokenization ideas can happen very quickly. Equally importantly, the repo is in entirely in python to make it easier for the greatest number of people to try out new ideas.

## Two Available Tokenizers

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:

0. [batchbpe/base.py](batchbpe/base.py): Implements the `Tokenizer` class, which is the base class. It has encode/decode and save/load functionality, and also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
1. [batchbpe/batch.py](batchbpe/batch.py): Implements the `BatchTokenizer` which includes a `train` method and `get_stats` and `merge_batch` functions needed to be able to train a new token vocabulary given input text. It inherits all the essentials from the `Tokenizer`.
2. [batchbpe/quick.py](batchbpe/quick.py): Implements the `QuickTokenizer` which is a small speed optimization of the `BatchTokenizer`. It runs ~8% faster by disregarding the issue of overcounting potential merges in sequences of repeated characters (e.g. "aaaaa" counts as only 2 possible "a"-"a" merges in the `BatchTokenizer`, but 4 in the `QuickTokenizer`), and by combining the `get_stats` and `merge_batch` functions into a single function. More importantly, the `QuickTokenizer` serves as a demonstration of how to implement your own new tokenizer that inherits from `Tokenizer` to test out a new tokenization approach or idea.

Finally, the script [train.py](train.py) trains the three major tokenizers on the input text [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her) and saves the vocab to disk for visualization. This script runs in about 25 seconds on my (M1) MacBook.

## Installation

1. Clone this repo: `git clone https://github.com/alexandermorgan/BatchBPE.git`
2. cd into batchbpe directory, set up a virtual environment and activate it
3. `uv pip install requirements.txt`

## Quick Start

For an example that's easy to inspect, we can reproduce the [Wikipedia example on BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) as follows:

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)
```

According to Wikipedia, running bpe on the input string: "aaabdaaabac" for 3 merges results in the string: "XdXac" where  X=ZY, Y=ab, and Z=aa. The tricky thing to note is that BatchBPE always allocates the 256 individual bytes as tokens, and then merges bytes as needed from there. So for us a=97, b=98, c=99, d=100 (their [ASCII](https://www.asciitable.com) values). Then when (a,a) is merged to Z, Z will become 256. Likewise Y will become 257 and X 258. So we start with the 256 bytes, and do 3 merges to get to the result above, with the expected output of [258, 100, 258, 97, 99].

This example also demonstrates that the batch merging approach to tokenization safely merges token pairs in batches. So in this highly artificial example, it makes the merges in the right order despite the fact that token 257 includes 256, and token 258 includes 257.

## Accepted Import Types

Data loading is the same for the Batch and Quick tokenizers. You can load text data via:

- literal string (as in the Wikipedia example above)
- path or url to .txt file
- datasets library Dataset object
- datasets library IterableDataset object (not recommended)
- dictionary or path to a json file of a dictionary saved by a previous Tokenizer data import
- list/tuple of any combination of the above

HuggingFace's datasets library is great way to get large specialized text datasets, though it can be a little tricky to use. When passing a regular Dataset object, BatchBPE will try to multiprocess the data loading using joblib. This default can be disabled by passing `multiprocess=False` when instantiating your tokenizer. Here's an example of passing a `datasets.Dataset` from a local parquet file.

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer()
path = 'path_to_parquet_file'
kwargs = {'path': 'parquet', 'data_files': {'train': path}, 'split': 'train'}
data = load_dataset(**kwargs)
tokenizer.train(data, 50304, verbose=True)
```

Alternatively this could be a `pyarrow.ChunkedArray` object like this:

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer()
path = 'full_path_to_parquet_file'
kwargs = {'path': 'parquet', 'data_files': {'train': path}, 'split': 'train'}
data = load_dataset(**kwargs).data['text']
tokenizer.train(data, 50304, verbose=True)
```

It is also possible to pass a `datasets.IterableDataset` though this is currently not recommended because it is very slow for large datasets. I think I must be processing it poorly so I hope to improve this. Here is the example:

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer()
path = 'full_path_to_parquet_file'
kwargs = {'path': 'melikocki/preprocessed_shakespeare', 'split': 'train', 'streaming': True}
data = load_dataset(**kwargs)
tokenizer.train(data, 50304, verbose=True)
```

Instead of streaming datasets, if you want to load datasets that don't fit in memory and/or hard-disk space, I recommend downloading as many of the files of the dataset at a time as possible, then converting those to BatchBPE's json representation of a dataset which is a greater than 100X compression for large datasets, and then deleting the dataset files. Do this for as many groups of files as necessary, then load a list of the json files BatchBPE produced to combine those. For example, the 10B token sample of the FineWeb-Edu dataset is about 27GB spread out among 10 files. Say you only have room for 5 of those files at a time on your computer, you could load the entire dataset in the following way. While this requires a bit more setup, subsequent uses of the json file will be very fast, so you only have to do it once.

```python
# download 5 files from the dataset before beginning
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer(store_dict=True)   # note `store_dict` param
paths = ['path_to_parquet_file_1', ... 'path_to_parquet_file_5']
kwargs = [{'path': 'parquet', 'data_files': {'train': path}, 'split': 'train'} for path in paths]
data = [load_dataset(**kwg) for kwg in kwargs]
tokenizer.train(data, 256)   # calling train with vocab_size of 256 will do no merges, but does call _import_data
# the json file will be saved in the format '{date}-{time}-dataset-dict.json'

# Then delete the five files, download the remaining dataset files and repeat the above with those files.

# Then import the two json file dataset distillations you have. You can also further compress this into one json file for future use.
tokenizer3 = BatchTokenizer(store_dict=True)
paths = ['path_to_first_json_file.json', 'path_to_second_json_file.json']
tokenizer3.train(paths, 50304)   # now you can train a vocabulary on your full dataset
# for later use, the data from all ten files combined will be in the new json file in the same '{date}-{time}-dataset-dict.json' format.
```

## Registering Special Tokens

The ability to register special tokens remains in tact from Karpathy's minbpe repo. Note that just like tiktoken, we have to explicitly declare our intent to use and parse special tokens in the call to encode. Otherwise this can become a major footgun, unintentionally tokenizing attacker-controlled data (e.g. user prompts) with special tokens. The `allowed_special` parameter can be set to "all", "none", or a list of special tokens to allow.

```python
from batchbpe import BatchTokenizer
text = "<|endoftext|>hello world"
specials = {'<|endoftext|>': 50304}
tokenizer = BatchTokenizer()
tokenizer.load('path_to_experiment_model_file')   # load existing vocab to save time
tokenizer.register_special_tokens(specials)

# still encodes without using the special tokens by default
print(tokenizer.encode(text))
# [100257, 15339, 1917]

# explicitly set allowed_special='all' to use special tokens
print(tokenizer.encode(text, allowed_special='all'))
# [100257, 15339, 1917]
```

## Special Features

### Batching Token Pair Merges

If the idea of batching token merges still makes you nervous, you can set `max_batch_size=1` to merge one pair at a time:

```python
tokenizer = BatchTokenizer()
path = 'path_to_first_json_file.json'
tokenizer.train(path, 50304, max_batch_size=1)
```

### Compress Text into Dictionary Representation

### Stop Word Handling

"Stop words" are the most common words used in a language and a "stop list" is a list of stop words. In the context of these tokenizers, they are more like "stop text chunks". When text is processed by the BatchBPE tokenizers, it gets split according to the split_pattern you use (default is GPT4 split pattern). The X most common of these chunks can be assigned individual tokens before the normal BPE vocabulary building process begins. This will assign special tokens to common 2+ character text chunks like " the", " in" ".\n", etc. BatchBPE automates this by letting you pass the `stop_list_size` param to the tokenizer on instantiation. Like this:

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer(stop_list_size=100)
data = 'full_path_to_json_file'
tokenizer.train(data, 50304, verbose=True)
```

Why would you do this when the most common text chunks already get represented by their own tokens in the normal BPE vocabulary building process? The distribution of text chunk frequencies overwhelming favors these stop words, so if you don't handle them explicitly, then the token merges will begin on a path that caters to them exclusively. Since earlier merges impact later merges, this may not be the most effective way to shape the entire vocabulary.

For example, consider the text chunk " in". As a standalone word, " in" describes location or membership. But as a prefix, " in" usually indicates negation which is close to the opposite of the stop word " in". Since the stop list tokens are only applied to text chunks if they match the entire text chunk, you can easily avoid this semantic pollution from the stop words. Another token will eventually get created that also points to the characters " in", but that one will be free to participate in further merges inside text chunks like " inaccessible". The `stop_list_size` feature lets you dynamically apply this approach to the X most common multi-character text chunks in your dataset. This effectively allows for a mix of word-level and byte-level encoding. Just don't get too carried away with the `stop_list_size` since it eats into your vocabulary size.

### Frequency Cutoff

The `freq_cutoff` parameter addresses the opposite problem: a very high percentage of text chunks in a dataset only occur once or a handful of times. If you want to enforce a threshhold for the number of times a text chunk must appear for it to participate in regular token pair merges, you can do this with `freq_cutoff`. The default value is 1 (i.e. all text chunks are considered) but if you set `freq_cutoff=2`, a text chunk would have to appear at least twice to be considered. This alone can eliminate over half of the unique text chunks in the dataset (obviously highly dependent on dataset) making training twice as fast. More importantly, all that noise from tons of non-repeating text chunks may actually make your tokenization worse. With BatchBPE it's easy to see that applying this kind of threshhold changes the course of the tokenization, but whether or not that change is for the better is up for experimentation.

The option to have a frequency threshhold follows naturally from the architectural decision to internally represent datasets with a dictionary mapping text chunks to their counts. You apply the parameter at the tokenizer instantiation stage like this:

```python
from batchbpe import BatchTokenizer
tokenizer = BatchTokenizer(freq_cutoff=2)
data = 'full_path_to_json_file'
tokenizer.train(data, 50304, verbose=True)
```

## Tests

We use the pytest library for tests. All of them are located in the `tests/` directory. First `uv pip install pytest` if you haven't already, then:

```bash
$ pytest -v .
```

to run the tests. (-v is verbose, slightly prettier).

## License

MIT
