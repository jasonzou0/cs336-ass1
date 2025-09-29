#!/usr/bin/env python3
"""
Create a proper BPE vocabulary with all 256 base bytes preserved
"""
import json
import os

def create_complete_vocab():
    """Create a vocabulary with all 256 base bytes + special token + some common merges"""
    
    vocab = {}
    
    # Add all 256 base bytes (0-255) using Latin-1 encoding to preserve exact bytes
    for i in range(256):
        # Use Latin-1 encoding which has 1-to-1 mapping with bytes 0-255
        vocab[str(i)] = bytes([i]).decode('latin-1')
    
    # Add special token
    vocab["256"] = "<|endoftext|>"
    
    # Add some common token merges (you could add more sophisticated ones)
    common_merges = [
        b" t", b"he", b" a", b" s", b"in", b"er", b"nd", b"ed", b" w", b" h",
        b"it", b"on", b"re", b"at", b"en", b"or", b"an", b"al", b"es", b"ng"
    ]
    
    next_id = 257
    for merge in common_merges:
        if next_id < 300:  # Just add a few for testing
            vocab[str(next_id)] = merge.decode('utf-8', errors='replace')
            next_id += 1
    
    return vocab

def create_basic_merges():
    """Create basic merge rules"""
    merges = [
        " t", "h e", " a", " s", "i n", "e r", "n d", "e d", " w", " h",
        "i t", "o n", "r e", "a t", "e n", "o r", "a n", "a l", "e s", "n g"
    ]
    return merges

def main():
    print('Creating proper BPE vocabulary with all base bytes...')
    
    # Create output directory
    os.makedirs('data/tokenizer', exist_ok=True)
    
    # Create vocabulary with all base bytes
    vocab = create_complete_vocab()
    
    # Save vocabulary
    with open('data/tokenizer/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save basic merges
    merges = create_basic_merges()
    with open('data/tokenizer/merges.txt', 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(f'{merge}\n')
    
    print(f'SUCCESS: Created BPE vocabulary!')
    print(f'Vocabulary size: {len(vocab)}')
    print(f'All base bytes included: {all(str(i) in vocab for i in range(256))}')
    print(f'Number of merges: {len(merges)}')
    print('Files saved to data/tokenizer/')
    
    # Verify by checking problematic byte
    print(f'\\nByte 226 (\\xe2) maps to: {repr(vocab["226"])}')

if __name__ == '__main__':
    main()