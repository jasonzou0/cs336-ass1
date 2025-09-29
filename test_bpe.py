#!/usr/bin/env python3
"""
Quick BPE test with small sample
"""
from cs336_basics.my_trainer import run_train_bpe
import json
import os

print('Testing BPE on small sample...')

# Use small sample file for quick testing
vocab, merges = run_train_bpe(
    input_path='tests/fixtures/tinystories_sample.txt',
    vocab_size=2000,
    special_tokens=['<|endoftext|>']
)

# Create output directory
os.makedirs('data/tokenizer', exist_ok=True)

# Save vocabulary
vocab_dict = {str(k): v.decode('utf-8', errors='replace') for k, v in vocab.items()}
with open('data/tokenizer/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

# Save merges
with open('data/tokenizer/merges.txt', 'w', encoding='utf-8') as f:
    for t1, t2 in merges:
        line = f'{t1.decode("utf-8", errors="replace")} {t2.decode("utf-8", errors="replace")}\n'
        f.write(line)

print(f'SUCCESS: Generated BPE tokenizer!')
print(f'Vocabulary size: {len(vocab)}')
print(f'Number of merges: {len(merges)}')
print('Files saved to data/tokenizer/')