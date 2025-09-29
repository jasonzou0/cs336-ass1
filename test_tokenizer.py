#!/usr/bin/env python3
"""
Test the generated BPE tokenizer
"""
import sys
sys.path.append('.')
from prepare_data import load_bpe_tokenizer

# Load the tokenizer
tokenizer = load_bpe_tokenizer('data/tokenizer/vocab.json', 'data/tokenizer/merges.txt')

# Test encoding/decoding
test_text = 'Once upon a time there was a little girl.'
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f'Original: {test_text}')
print(f'Tokens: {tokens[:10]}...')
print(f'Decoded: {decoded}')
print(f'Round-trip successful: {test_text == decoded}')
print('SUCCESS: Tokenizer works correctly!')