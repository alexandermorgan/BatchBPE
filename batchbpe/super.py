"""
"Super" byte pair encoding tokenizer. Merges are made in the standard
way until merge T at which point merges will be allowed between any
two adjacent tokens, even if they would have been excluded from mreges
given the split pattern used in the initial standard BPE phase.
"""
import numpy as np
import os
import shutil
import time
from .base import Tokenizer
from .batch import get_stats as p1_get_stats
from collections import defaultdict
from heapq import nlargest
# from datasets import load_dataset, IterableDataset, Dataset
from concurrent.futures import ThreadPoolExecutor
import psutil

def p1_merge_batch_and_get_stats(ids, pairs):
    counts = defaultdict(int)
    for chunk, num in ids:
        last_index = len(chunk) - 1
        i = 0
        # two_back is used to prevent overcounting consective pairs
        # e.g. 'aaaaa' should only increase the a-a count by 2, not 4.
        two_back = None
        while i < last_index:
            # the merging part
            j = i + 1
            token = pairs.get((chunk[i], chunk[j]))
            if token is not None:
                chunk[i] = token
                del chunk[j]
                last_index -= 1

            # the get_stats counting part
            # we look back one token when counting in case the current token
            # changed because of a merge, so make sure that i != 0
            if i:
                if chunk[i] == chunk[i-1]:
                    if chunk[i] != two_back:  # i and i-1 are the same token, but two_back is None
                        counts[(chunk[i-1], chunk[i])] += num
                        two_back = chunk[i]
                    else:  # 3 of the same token in a row so don't add to counts
                        two_back = None
                else:
                    counts[(chunk[i-1], chunk[i])] += num
                    two_back = None
            i = j
        if (i and i == last_index  # the last pair in chunk did not merge
            and not (chunk[-2] == chunk[-1] == two_back)): # this is not a repetition overcounting case
            counts[(chunk[-2], chunk[-1])] += num
    return counts

def p2_merge_batch_and_get_stats(pairs):
    counts = defaultdict(int)
    doc_paths = [f"temp.noindex/{doc}" for doc in os.listdir("temp.noindex")]
    n_jobs = min(os.cpu_count() or 1, 3)
    docs_per_job = len(doc_paths) // n_jobs + 1
    
    # Process documents in parallel batches
    def process_batch(batch_paths, batch_counts=None):
        # use the provided counts for the first worker, create new one for others
        if batch_counts is None:
            batch_counts = defaultdict(int)
        
        for path in batch_paths:
            ids = np.load(path)
            merged_ids = np.empty_like(ids)
            last_index = len(ids) - 1
            num_merges = 0
            _g, _h = None, None
            i = 0
            while i < last_index:
                j = i + 1
                if (token := pairs.get((ids[i], ids[j]))) is not None:
                    merged_ids[i - num_merges] = token
                    num_merges += 1
                    i += 2
                else:
                    token = ids[i]
                    merged_ids[i - num_merges] = token
                    i = j
                
                # count pairs
                if _h is not None:
                    if not (_g == _h == token):
                        batch_counts[(_h, token)] += 1
                        _g = _h
                    else:
                        _g = None
                _h = token
            
            # handle last token if it wasn't merged away
            if i and i == last_index and not (_g == _h == ids[i]):
                batch_counts[(_h, ids[i])] += 1
                merged_ids[i - num_merges] = ids[i]
            
            to_write = merged_ids[:last_index - num_merges + 1]
            if len(to_write) < 2:
                # delete the file
                os.remove(path)
            elif num_merges:  # save result if there were merges
                np.save(path, to_write)
        
        return batch_counts
    
    # Create batches and process in parallel
    batches = [doc_paths[i:i+docs_per_job] for i in range(0, len(doc_paths), docs_per_job)]
    
    # Use a smaller chunksize for better load balancing
    # Prepare arguments: first batch gets the counts dict, others get None
    batch_args = [(batch, counts if i == 0 else None) for i, batch in enumerate(batches)]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(lambda args: process_batch(*args), batch_args))
    
    # Combine counts from all batches (skip the first one since it's already in counts)
    combine_start_time = time.time()
    for batch_counts in results[1:]:
        for pair, count in batch_counts.items():
            counts[pair] += count
    combine_end_time = time.time()
    print(f"Time to combine counts from all batches: {combine_end_time - combine_start_time:.4f} seconds")
    
    return counts

def _p2_get_stats_helper(counts: defaultdict[tuple[int, int]: int], 
                         ids: np.array, filename: str) -> None:
    """
    Do the actual counting of token pairs from the `ids` and add them to what's
    already in the `counts` dict. This is separate from SuperTokenizer.p2_get_stats
    because some filetypes have just one text document (like a .txt file) but others
    contain multiple documents (like a .parquet file). The `counts` defaultdict is
    modified in place so nothing is returned.
    """
    # iterate over ids adding each pair to counts
    last_index = len(ids) - 1
    i = 0
    while i < last_index:
        j = i + 1
        counts[(ids[i], ids[j])] += 1
        if ids[i] == ids[j] and j+1 <= last_index and ids[i] == ids[j+1]:
            i += 2  # skip the next token to avoid overcounting consecutive repeated pairs
        else:
            i = j
    
    np.save(filename, ids)

def _p2_process_articles_batch(articles_batch, doc_index, start_index, encode_method, counts=None):
    """Process a batch of articles in parallel and return the counts dictionary."""
    if counts is None:
        counts = defaultdict(int)
    for i, article in enumerate(articles_batch):
        article_index = start_index + i
        ids = np.array(encode_method(article.as_py()), dtype=np.int32)
        _p2_get_stats_helper(counts, ids, f"temp.noindex/{doc_index}_{article_index}.npy")
    return counts

class SuperTokenizer(Tokenizer):
    def __init__(self, pattern=None, multiprocess=True, store_dict=False,
            stop_list_size=0, freq_cutoff=0):
        """
        TODO: docstring
        """
        if os.path.exists("temp.noindex"):
            shutil.rmtree("temp.noindex")
        os.makedirs("temp.noindex")  # temp dir to store intermediate phase 2 tokenization results
        super().__init__(pattern, multiprocess, store_dict, stop_list_size, freq_cutoff)

    def p2_get_stats(self, docs):
        """
        Convert all the data to numpy arrays of ints, count the pairs, and save 
        the numpy arrays to temp files for the next iteration in
        p2_merge_batch_and_get_stats.
        """
        counts = defaultdict(int)
        # for doc_index, doc in enumerate([docs]):
        #     if isinstance(doc, str) and doc.endswith('.txt'):
        #         print("Warning: Text files are not recommended for phase 2 parsing.\
        #               This will still work but consider using parquet files instead.")
        #         with open(doc, "r") as f:
        #             text = f.read()
        #         ids = np.array(self.encode(text), dtype=np.int32)
        #         _p2_get_stats_helper(counts, ids, f"temp.noindex/{doc_index}.npy")
        #     elif isinstance(doc, str) and os.path.isfile(doc) and doc.endswith('.parquet'):
        #         articles = load_dataset('parquet', data_files=doc).data['train'].flatten()[0]
        #         print(f'{len(articles)} articles in {doc}')
                
        #         # Parallelize processing with joblib
        #         n_jobs = 3  # cpu_count()
        #         print(f'Using {n_jobs} cores')
        #         article_limit = 100000
        #         articles = articles[:article_limit]
        #         batch_size = len(articles) // n_jobs + 1
                
        #         # Create batches
        #         batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
                
        #         # Prepare arguments: first batch gets the counts dict, others get None
        #         batch_args = [(batch, doc_index, i*batch_size, self.encode, counts if i == 0 else None) 
        #                      for i, batch in enumerate(batches)]
                
        #         # Process articles in parallel batches with modified function signature
        #         current_process = psutil.Process()
        #         subproc_before = set([p.pid for p in current_process.children(recursive=True)])
        #         results = Parallel(n_jobs=n_jobs, prefer="threads")(
        #             delayed(_p2_process_articles_batch)(*args)
        #             for args in batch_args
        #         )
        #         subproc_after = set([p.pid for p in current_process.children(recursive=True)])
        #         for subproc in subproc_after - subproc_before:
        #             print('Killing process with pid {}'.format(subproc))
        #             psutil.Process(subproc).terminate()
        #         # Combine counts from all batches (skip the first one since it's already in counts)
        #         for result in results[1:]:
        #             for pair, count in result.items():
        #                 counts[pair] += count
                        
        return counts

    def train(self, data, vocab_size, T=0.9, p1_data=None, verbose=False):
        # validate and calculate T as an integer
        if isinstance(T, int):
            if not 0 <= T <= vocab_size:
                print('T cannot be larger than the vocab size or less than 0. Try again.')
                return
            p1_vocab_size = T
        elif isinstance(T, float):
            if not 0 <= T <= 1:
                print('If T is a float, it must be between 0 and 1 inclusive. Try again.')
                return
            p1_vocab_size = round(T * vocab_size)
        else:
            print('T must be an int between 256 and the vocab_size or a float between 0 and 1 inclusive. Try again')
            return

        # phase 1 data load
        t0 = time.time()
        ids = self._import_data(p1_data or data)   # [(list_of_int_tokens, int)] -> text chunks and their counts
        t1 = time.time()
        print(f'Time spent loading data for phase 1: {t1-t0:.2f}')

        merges = self.merges   # {(int, int): int} -> token pair to new token
        vocab = self.vocab   # {int: bytes} -> token to its bytes representation
        batch_count = 0
        curr_vocab_size = len(vocab) + len(self.special_tokens)
        stats = p1_get_stats(ids)  # stats are later updated by p1_merge_batch_and_get_stats
        t1 = time.time()
        if verbose:
            print(f'Time spent getting initial token-pair counts for phase 1: {t1-t0:.2f}')

        for p in (1, 2):  # phases 1 and 2
            t0 = time.time()
            if p == 1:
                stats = p1_get_stats(ids)
                merges_remaining = p1_vocab_size - curr_vocab_size
            else:
                stats = self.p2_get_stats(data)
                merges_remaining = vocab_size - curr_vocab_size
            t1 = time.time()
            if verbose:
                print(f'Time spent getting initial token-pair counts for phase {p}: {t1-t0:.2f}')
            while merges_remaining:
                seen_first = set()   # tokens seen in the first position in pairs
                seen_last = set()   # tokens seen in the last position in pairs
                pairs_to_merge = {}
                num_pairs_to_search = min(merges_remaining//2, len(vocab)) or 1  # or 1 because it would be 0 if merges_remaining is 1
                top_pairs = nlargest(num_pairs_to_search, stats, key=stats.get)
                for first, last in top_pairs:  # pairs are (first, last) tuples
                    if first in seen_last or last in seen_first:   # unsafe merge
                        seen_first.add(first)
                        seen_last.add(last)
                        continue # skip this pair but keep looking for safe merges in top_pairs
                    seen_first.add(first)
                    seen_last.add(last)
                    pairs_to_merge[(first, last)] = curr_vocab_size
                    try:
                        vocab[curr_vocab_size] = vocab[first] + vocab[last]
                    except:
                        import pdb; pdb.set_trace()
                    curr_vocab_size += 1
                merges_remaining -= len(pairs_to_merge)
                merges.update(pairs_to_merge)  # save the merges
                batch_count += 1
                if merges_remaining:   # no need to merge last batch
                    if p == 1:
                        stats = p1_merge_batch_and_get_stats(ids, pairs_to_merge)   # replace pairs_to_merge keys in ids with their values
                    else:
                        stats = p2_merge_batch_and_get_stats(pairs_to_merge)
                if verbose:
                    t2 = time.time()
                    print(f"Batch {batch_count} merged {len(pairs_to_merge)} pairs in {t2-t1:.2f} sec. Phase {p} merges left: {merges_remaining}")
                    t1 = t2
            if p == 1:
                self.p1_batch_count = batch_count
                # also delete ids
        self.p2_batch_count = batch_count - self.p1_batch_count
        if os.path.exists("temp.noindex"):
            shutil.rmtree("temp.noindex")
