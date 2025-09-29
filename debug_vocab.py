#!/usr/bin/env python3
"""
Debug vocabulary structure
"""
import json

with open('data/tokenizer/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

print('Checking tokens at specific positions:')
for i in [127, 128, 129, 130, 255, 256, 257]:
    if str(i) in vocab:
        token = vocab[str(i)]
        byte_repr = repr(token.encode('utf-8'))
        print(f'ID {i}: {repr(token)} -> {byte_repr}')
    else:
        print(f'ID {i}: NOT FOUND')

print(f'Max token ID: {max(int(k) for k in vocab.keys())}')

# Check how many single-byte tokens we have at the start
single_byte_count = 0
for i in range(256):
    if str(i) in vocab:
        token = vocab[str(i)]
        if len(token.encode('utf-8')) == 1:
            single_byte_count += 1
        else:
            print(f'ID {i} is not single byte: {repr(token)}')
            break
    else:
        print(f'Missing token at position {i}')
        break

print(f'Consecutive single bytes from start: {single_byte_count}')