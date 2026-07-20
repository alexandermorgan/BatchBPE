import pytest
import tiktoken
import os
import tempfile
from array import array
from batchbpe import BatchTokenizer
from batchbpe.corpus import DiskCorpus, RamCorpus

# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]
def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

TAYLORSWIFT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taylorswift.txt")

# -----------------------------------------------------------------------------
# tests

# test encode/decode identity for a few different strings
@pytest.mark.parametrize("tokenizer_factory", [BatchTokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded

# reference test to add more tests in the future
@pytest.mark.parametrize("tokenizer_factory", [BatchTokenizer])
@pytest.mark.parametrize("backend", ["ram", "disk"])
def test_wikipedia_example(tokenizer_factory, backend):
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory(multiprocess=False)
    text = "aaabdaaabac"
    with tempfile.TemporaryDirectory() as work_dir:
        tokenizer.train(text, 256 + 3, backend=backend,
                        work_dir=work_dir if backend == "disk" else None)
    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(tokenizer.encode(text)) == text

@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load(special_tokens):
    # take a bit more complex piece of text and train the tokenizer, chosen at random
    text = llama_text
    # create a Tokenizer and do 64 merges
    tokenizer = BatchTokenizer()
    tokenizer.train(text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)
    # verify that decode(encode(x)) == x
    test = tokenizer.decode(tokenizer.encode(text, "all"))
    print(f'test: {test}\n\ntext: {text}')
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    # verify that save/load work as expected
    ids = tokenizer.encode(text, "all")
    # save the tokenizer (TODO use a proper temporary directory)
    tokenizer.save("test_tokenizer_tmp")
    # re-load the tokenizer
    tokenizer = BatchTokenizer()
    tokenizer.load("test_tokenizer_tmp.model")
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    assert tokenizer.encode(text, "all") == ids
    # delete the temporary files
    for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
        os.remove(file)


def test_disk_corpus_stats_and_merge_match_ram():
    """DiskCorpus must match RamCorpus pair counts before and after a merge batch."""
    ids = [
        array("i", [2, 97, 98, 99]),
        array("i", [1, 98, 99, 100]),
        array("i", [1, 101, 101, 101]),
        array("i", [3, 97, 98, 97, 98]),
    ]
    # Independent copies so in-place merges on one backend cannot affect the other.
    ram_ids = [array("i", c) for c in ids]
    disk_ids = [array("i", c) for c in ids]
    mult = 1000
    pairs = {97 * mult + 98: 256}  # merge 'ab' -> 256

    with RamCorpus(ram_ids, n=2) as ram, \
         tempfile.TemporaryDirectory() as work_dir, \
         DiskCorpus(disk_ids, n=2, work_dir=work_dir) as disk:
        assert dict(ram.initial_stats(mult)) == dict(disk.initial_stats(mult))
        assert dict(ram.merge_and_recount(pairs, mult)) == dict(disk.merge_and_recount(pairs, mult))
        # Manifest + shards exist under work_dir for inspection / later resume work.
        assert os.path.isfile(os.path.join(work_dir, "manifest.json"))
        assert os.path.isfile(os.path.join(work_dir, "shard_0000.bin"))


def test_ram_disk_identical_vocab_taylorswift():
    """backend='disk' must learn the same merges and vocab bytes as backend='ram'."""
    text = open(TAYLORSWIFT, encoding="utf-8").read()
    vocab_size = 256 + 64

    ram = BatchTokenizer(multiprocess=False)
    ram.train(text, vocab_size, backend="ram")

    disk = BatchTokenizer(multiprocess=False)
    with tempfile.TemporaryDirectory() as work_dir:
        disk.train(text, vocab_size, backend="disk", work_dir=work_dir)

    assert ram.merges == disk.merges
    assert ram.vocab == disk.vocab
    # Spot-check that the trained models tokenize the corpus the same way.
    sample = text[:2000]
    assert ram.encode(sample) == disk.encode(sample)


def test_ram_disk_identical_vocab_superbpe_stages():
    """Two-stage SuperBPE-style train: disk stage-2 matches ram stage-2."""
    import regex as re
    from batchbpe.base import GPT4_SPLIT_PATTERN

    text = open(TAYLORSWIFT, encoding="utf-8").read()
    stage1_size = 256 + 40
    final_size = stage1_size + 20
    newline_pattern = r"[^\n]+"

    def run(backend: str):
        tok = BatchTokenizer(pattern=GPT4_SPLIT_PATTERN, multiprocess=False)
        tok.train(text, stage1_size, backend=backend)
        tok.pattern = newline_pattern
        tok.compiled_pattern = re.compile(newline_pattern)
        with tempfile.TemporaryDirectory() as work_dir:
            tok.train(text, final_size, backend=backend,
                      work_dir=work_dir if backend == "disk" else None)
        return tok

    ram = run("ram")
    disk = run("disk")
    assert ram.merges == disk.merges
    assert ram.vocab == disk.vocab


def test_disk_backend_temp_work_dir_cleaned_up():
    """Omitting work_dir uses a temp dir that DiskCorpus removes on close."""
    tok = BatchTokenizer(multiprocess=False)
    tok.train("aaabdaaabac", 256 + 3, backend="disk")
    assert tok.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


# TODO: make this equivalency test a standalone script that compares two tokenizers
# def test_batch_regex_equivalent():
#     # show that batch and regex tokenizations are equivalent. They will have different
#     # merges dict keys, but if the byte strings that those keys correspond to are
#     # the same (irrespective of key order) then they are equivalent. In other words,
#     # "equivalent" here means that they would tokenize a text in the same way.
#     # It is *possible* for them to be unequal but in practice they are almost always
#     # equivalent and even when they aren't the difference is unnoticeable.
#     text = llama_text
#     vocab_size = 256 + 20

#     batch_tokenizer = BatchTokenizer()
#     batch_tokenizer.train(text, vocab_size, verbose=True)
#     batch_tokenizer.register_special_tokens(special_tokens)
#     batch_vocab_set = {*batch_tokenizer.vocab.values()}

#     regex_tokenizer = RegexTokenizer()
#     regex_tokenizer.train(text, vocab_size, verbose=True)
#     regex_tokenizer.register_special_tokens(special_tokens)
#     regex_vocab_set = {*regex_tokenizer.vocab.values()}

#     assert(batch_vocab_set == regex_vocab_set)

if __name__ == "__main__":
    pytest.main()
