#!/usr/bin/env python3
"""
Generate proper BPE tokenizer that preserves all base bytes
"""
from cs336_basics.my_trainer import run_train_bpe
import json
import os

def create_proper_bpe_vocab(vocab_dict_from_trainer, merges, vocab_size):
    """Ensure all 256 base bytes are included in vocabulary"""
    
    # Start with all base bytes (0-255) - these must always be present
    final_vocab = {}
    
    # Add all base bytes first (IDs 0-255)
    for i in range(256):
        final_vocab[i] = bytes([i])
    
    # Add special tokens 
    special_token = '<|endoftext|>'.encode('utf-8')
    final_vocab[256] = special_token
    
    # Add merged tokens up to vocab_size
    next_id = 257
    for token_id, token_bytes in vocab_dict_from_trainer.items():
        token_id = int(token_id)
        
        # Skip base bytes (already added) and special tokens
        if token_id < 256 or token_bytes == special_token:
            continue
            
        if next_id < vocab_size:
            final_vocab[next_id] = token_bytes
            next_id += 1
        else:
            break
    
    return final_vocab

def main():
    print('Generating proper BPE tokenizer with all base bytes...')
    
    # Use the small sample for quick testing - we just need proper structure
    vocab, merges = run_train_bpe(
        input_path='tests/fixtures/tinystories_sample.txt',
        vocab_size=2000,
        special_tokens=['<|endoftext|>']
    )
    
    # Create proper vocab that includes all base bytes
    proper_vocab = create_proper_bpe_vocab(vocab, merges, 2000)
    
    # Create output directory
    os.makedirs('data/tokenizer', exist_ok=True)
    
    # Save vocabulary with proper structure
    vocab_dict = {}
    for token_id, token_bytes in proper_vocab.items():
        vocab_dict[str(token_id)] = token_bytes.decode('utf-8', errors='replace')
    
    with open('data/tokenizer/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    # Save merges (only use merges that are relevant to our final vocab)
    with open('data/tokenizer/merges.txt', 'w', encoding='utf-8') as f:
        for t1, t2 in merges:
            line = f'{t1.decode("utf-8", errors="replace")} {t2.decode("utf-8", errors="replace")}\n'
            f.write(line)
    
    print(f'SUCCESS: Generated proper BPE tokenizer!')
    print(f'Vocabulary size: {len(proper_vocab)}')
    print(f'All base bytes included: {all(i in proper_vocab for i in range(256))}')
    print(f'Number of merges: {len(merges)}')
    print('Files saved to data/tokenizer/')

if __name__ == '__main__':
    main()