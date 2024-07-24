# BatchBPE

Practical performant pure python implementation of the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings. "Batch" is for the most characteristic aspect of BatchBPE, which is that it executes token-pair merges safely in batches. ~200 is the average batch size when building a ~50k token vocabulary.

This repo is a fork of Andrej Karpathy's excellent introduction to the BPE used for LLM tokenization. If you're new to the subject but haven't reviewed Karpathy's resources, definitely start there. He has a [2-hour video](), accompanying [minbpe github repo], and a [colab notebook](). I highly recommend all three. Tokenization is deceptively simple, so a deep dive into the topic is definitely worth it even if you can understand the basics of it with a 60-second description.

This BatchBPE began as a PR for minbpe but developed to the point where the objective changed. Instead of minbpe's pedagogic purpose, BatchBPE aims to be as practical and easy to modify as possible. The goal is to make it easy for people to try out new tokenization ideas even if they're working with limited compute, memory, or hard-disk resources. A lot of making tokenization more accessible boils down to compute and memory optimizations. Using BatchBPE's fastest combination of settings (described below), you can train a GPT2-sized vocabulary (~50k tokens) on 1GB's worth of text in well under a minute on a 4-year old laptop. So yes, trying out new tokenization ideas can happen very quickly. Equally importantly, the repo is in entirely in python to make it easier for the greatest number of people to try out new ideas.

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:

0. [batchbpe/base.py](batchbpe/base.py): Implements the `Tokenizer` class, which is the base class. It has encode/decode and save/load functionality, and also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
1. [batchbpe/batch.py](batchbpe/batch.py): Implements the `BatchTokenizer` which includes a `train` method and `get_stats` and `merge_batch` functions needed to be able to train a new token vocabulary given input text. It inherits all the essentials from the `Tokenizer`.
2. [batchbpe/quick.py](batchbpe/quick.py): Implements the `QuickTokenizer` which is a small speed optimization of the `BatchTokenizer`. It runs ~8% faster by disregarding the issue of overcounting potential merges in sequences of repeated characters (e.g. "aaaaa" counts as only 2 possible "a"-"a" merges in the `BatchTokenizer`, but 4 in the `QuickTokenizer`), and by combining the `get_stats` and `merge_batch` functions into a single function. More importantly, the `QuickTokenizer` serves as a demonstration of how to implement your own new tokenizer that inherits from `Tokenizer` to test out a new tokenization approach or idea.

Finally, the script [train.py](train.py) trains the three major tokenizers on the input text [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her) and saves the vocab to disk for visualization. This script runs in about 25 seconds on my (M1) MacBook.

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file.

## quick start

As the simplest demonstration, we can reproduce the [Wikipedia example on BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) as follows:

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

This example also demonstrates that the `BatchTokenizer` safely merges token pairs in batches. So in this highly artificial example, it makes the merges in the right order despite the fact that token 257 includes 256, and token 258 includes 257.

## inference: GPT-4 comparison

We can verify that the `RegexTokenizer` has feature parity with the GPT-4 tokenizer from [tiktoken](https://github.com/openai/tiktoken) as follows:

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

(you'll have to `pip install tiktoken` to run). Under the hood, the `GPT4Tokenizer` is just a light wrapper around `RegexTokenizer`, passing in the merges and the special tokens of GPT-4. We can also ensure the special tokens are handled correctly:

```python
text = "<|endoftext|>hello world"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# [100257, 15339, 1917]
```

Note that just like tiktoken, we have to explicitly declare our intent to use and parse special tokens in the call to encode. Otherwise this can become a major footgun, unintentionally tokenizing attacker-controlled data (e.g. user prompts) with special tokens. The `allowed_special` parameter can be set to "all", "none", or a list of special tokens to allow.

## training

Unlike tiktoken, this code allows you to train your own tokenizer. In principle and to my knowledge, if you train the `RegexTokenizer` on a large dataset with a vocabulary size of 100K, you would reproduce the GPT-4 tokenizer.

There are three paths you can follow. First, you can decide that you don't want the complexity of splitting and preprocessing text with regex patterns, and you also don't care for special tokens. In that case, reach for the `BasicTokenizer`. You can train it, and then encode and decode for example as follows:

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("mymodel") # writes mymodel.model and mymodel.vocab
tokenizer.load("mymodel.model") # loads the model back, the vocab is just for vis
```

If you instead want to follow along with OpenAI did for their text tokenizer, it's a good idea to adopt their approach of using regex pattern to split the text by categories. The GPT-4 pattern is a default with the `RegexTokenizer`, so you'd simple do something like:

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("tok32k") # writes tok32k.model and tok32k.vocab
tokenizer.load("tok32k.model") # loads the model back from disk
```

Where, of course, you'd want to change around the vocabulary size depending on the size of your dataset.

Once you're comfortable with the Basic and Regex tokenizers, you may want to use the `BatchTokenizer` which is an optimization of the `RegexTokenizer`. The regex handling is the same as for the `RegexTokenizer`:

```python
from minbpe import BatchTokenizer
tokenizer = BatchTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("tok32k") # writes tok32k.model and tok32k.vocab
tokenizer.load("tok32k.model") # loads the model back from disk
```

**Special tokens**. Finally, you might wish to add special tokens to your tokenizer. Register these using the `register_special_tokens` function. For example if you train with vocab_size of 32768, then the first 256 tokens are raw byte tokens, the next 32768-256 are merge tokens, and after those you can add the special tokens. The last "real" merge token will have id of 32767 (vocab_size - 1), so your first special token should come right after that, with an id of exactly 32768. So:

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

You can of course add more tokens after that as well, as you like. Finally, I'd like to stress that I tried hard to keep the code itself clean, readable and hackable. You should not have feel scared to read the code and understand how it works. The tests are also a nice place to look for more usage examples. That reminds me:

## tests

We use the pytest library for tests. All of them are located in the `tests/` directory. First `pip install pytest` if you haven't already, then:

```bash
$ pytest -v .
```

to run the tests. (-v is verbose, slightly prettier).

## community extensions

* [gnp/minbpe-rs](https://github.com/gnp/minbpe-rs): A Rust implementation of `minbpe` providing (near) one-to-one correspondence with the Python version

## exercise

For those trying to study BPE, here is the advised progression exercise for how you can build your own minbpe step by step. See [exercise.md](exercise.md).

## lecture

I built the code in this repository in this [YouTube video](https://www.youtube.com/watch?v=zduSFxRajkE). You can also find this lecture in text form in [lecture.md](lecture.md).

## todos

- write a more optimized Python version that could run over large files and big vocabs
- write an even more optimized C or Rust version (think through)
- rename GPT4Tokenizer to GPTTokenizer and support GPT-2/GPT-3/GPT-3.5 as well?
- write a LlamaTokenizer similar to GPT4Tokenizer (i.e. attempt sentencepiece equivalent)

## License

MIT
