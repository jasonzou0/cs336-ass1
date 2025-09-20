from einops import einsum, rearrange
import multiprocessing as mp

def apply_bpe(text, merges):
    # Dummy BPE implementation
    for merge in merges:
        text = text.replace(merge[0] + merge[1], merge[0] + '_' + merge[1])
    return text

def bpe_worker(chunk, bpe_merges):
    # Apply BPE merges to the chunk
    return [apply_bpe(line, bpe_merges) for line in chunk]

def multiprocess_bpe(input_lines, bpe_merges, num_workers=8):
    chunk_size = max(1,len(input_lines) // num_workers)
    chunks = [input_lines[i:i+chunk_size] for i in range(0, len(input_lines), chunk_size)]
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(bpe_worker, [(chunk, bpe_merges) for chunk in chunks])
    # Flatten results
    return [line for chunk in results for line in chunk]

def test_trainbpe_multiprocess():
    # Usage
    input_lines = ["hello world", 
                   "byte pair encoding", 
                   "multiprocessing example",
                   "hahahahaha","hohohohoho",
                   "heiheiheihiehei",
                   "neneenenene", 
                   "rorororororo",
                   "nononononono",
                   "rorororororo",
                   "nonononononasdfasf",
                   "sniffer is here",
                   "hohoheiheihei"]
    bpe_merges = [("e", "n"), ("o", "r"), ("h", "o"), ("h", "a"), ("i", "e")]

    encoded = multiprocess_bpe(input_lines, bpe_merges)
    print(encoded)

    assert encoded == ['hello wo_rld', 
                       'byte pair e_ncoding',
                       'multiprocessing example',
                       'h_ah_ah_ah_ah_a',
                       'h_oh_oh_oh_oh_o',
                       'heiheiheihi_ehei',
                       'ne_nee_ne_ne_ne',
                       'ro_ro_ro_ro_ro_ro',
                       'nononononono',
                       'ro_ro_ro_ro_ro_ro',
                       'nonononononasdfasf',
                       'sniffer is here',
                       'h_oh_oheiheihei']