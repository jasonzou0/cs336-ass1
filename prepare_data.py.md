01:28:50 leo@u18:/home/leo/workspace/Course/cs336/cs336-ass1 
$ uv run prepare_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/tinystories_custom.bin --vocab-size 32000 --method bpe       
Training BPE tokenizer with vocab_size=32000 on data/TinyStoriesV2-GPT4-train.txt
Training BPE with vocab_size=32000, special_tokens=['<|endoftext|>'], start_time=1758950942.3191524
Built vocab in 2361.115075826645 seconds
Trimmed/Padded vocab in 3.0517578125e-05 seconds
Error with BPE tokenization: cannot import name 'Tokenizer' from 'cs336_basics.my_tokenizer' (/home/leo/workspace/Course/cs336/cs336-ass1/cs336_basics/my_tokenizer.py)
Falling back to whitespace tokenization...
Using simple whitespace tokenization for data/TinyStoriesV2-GPT4-train.txt
Building vocabulary...
Counting words: 15600057it [00:58, 267740.16it/s]
Vocabulary size: 32000
Tokenizing...
Tokenizing: 15600057it [00:53, 293186.28it/s]
Saved 439,223,229 tokens to data/tinystories_custom.bin
Saved vocabulary to data/tinystories_custom_vocab.json

Success! Processed 439,223,229 tokens
Output file: data/tinystories_custom.bin
File size: 1675.5 MB
02:10:28 leo@u18:/home/leo/workspace/Course/cs336/cs336-ass1 
