#!/usr/bin/env python3
"""
Test tokenizer with problematic Unicode characters
"""
import sys
sys.path.append('.')
from prepare_data import load_bpe_tokenizer

# Load the tokenizer
tokenizer = load_bpe_tokenizer('data/tokenizer/vocab.json', 'data/tokenizer/merges.txt')

# Test with text that contains byte 0xe2 (common in Unicode sequences like em-dash —)
test_texts = [
    'Hello world!',
    'This has an em-dash — in it',  # em-dash contains \xe2\x80\x94
    'Café with accents',             # é contains \xc3\xa9  
    'Testing unicode: 🙂',           # emoji contains various bytes
]

for i, text in enumerate(test_texts):
    print(f'\\n--- Test {i+1}: {repr(text)} ---')
    try:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f'Tokens: {tokens[:15]}...')
        print(f'Decoded: {repr(decoded)}')
        print(f'Round-trip successful: {text == decoded}')
        
        # Show which bytes are in the original
        original_bytes = text.encode('utf-8')
        print(f'Original bytes: {[hex(b) for b in original_bytes[:20]]}...')
        
    except Exception as e:
        print(f'ERROR: {e}')

print('\\nOverall test completed!')