#!/usr/bin/env python3
"""
Simple BPE token generation script
"""
from cs336_basics.my_trainer import run_train_bpe
import json
import os

def main():
    print('Starting BPE training on TinyStories dataset...')
    
    # Train BPE tokenizer
    vocab, merges = run_train_bpe(
        input_path='data/TinyStoriesV2-GPT4-train.txt',
        vocab_size=8000,
        special_tokens=['<|endoftext|>']
    )
    
    # Create tokenizer directory
    os.makedirs('data/tokenizer', exist_ok=True)
    
    # Save vocabulary (convert bytes to strings for JSON serialization)
    vocab_dict = {}
    for token_id, token in vocab.items():
        try:
            # Use repr to handle special characters properly
            vocab_dict[str(token_id)] = token.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Could not decode token {token_id}: {e}")
            vocab_dict[str(token_id)] = repr(token)[2:-1]  # Remove b' and '
    
    with open('data/tokenizer/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    # Save merges (convert bytes to strings) 
    with open('data/tokenizer/merges.txt', 'w', encoding='utf-8') as f:
        for merge_pair in merges:
            try:
                token1, token2 = merge_pair
                token1_str = token1.decode('utf-8', errors='replace')
                token2_str = token2.decode('utf-8', errors='replace')
                f.write(f'{token1_str} {token2_str}\n')
            except Exception as e:
                print(f"Warning: Could not decode merge pair {merge_pair}: {e}")
                continue
    
    print(f'BPE training complete!')
    print(f'Vocabulary size: {len(vocab)}')
    print(f'Number of merges: {len(merges)}')
    print(f'Files saved to data/tokenizer/')

if __name__ == '__main__':
    main()