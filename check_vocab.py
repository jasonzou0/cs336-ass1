#!/usr/bin/env python3
"""
Check BPE vocabulary completeness
"""
import json

print('Checking BPE vocabulary...')

with open('data/tokenizer/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Check which base bytes are missing
present_bytes = set()
for token_id, token_str in vocab.items():
    if len(token_str.encode('utf-8')) == 1:
        byte_val = token_str.encode('utf-8')[0]
        present_bytes.add(byte_val)

missing_bytes = set(range(256)) - present_bytes
print(f'Total vocab size: {len(vocab)}')
print(f'Present single bytes: {len(present_bytes)}')
print(f'Missing bytes: {len(missing_bytes)}')

if missing_bytes:
    print(f'Missing byte values: {sorted(list(missing_bytes))[:20]}...')
    print('This explains the tokenization error!')
else:
    print('All base bytes are present - error must be elsewhere.')