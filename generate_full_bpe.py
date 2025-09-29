#!/usr/bin/env python3
"""
Generate BPE tokenizer using the full TinyStories dataset
"""
from cs336_basics.my_trainer import run_train_bpe
import json
import os

def main():
    print('Training BPE on full TinyStories dataset...')
    print('This may take several minutes...')
    
    # Train BPE tokenizer on the full dataset
    vocab, merges = run_train_bpe(
        input_path='data/TinyStoriesV2-GPT4-train.txt',
        vocab_size=8000,
        special_tokens=['<|endoftext|>']
    )
    
    # Create output directory
    os.makedirs('data/tokenizer', exist_ok=True)
    
    # Save vocabulary
    print(f'Saving vocabulary with {len(vocab)} tokens...')
    vocab_dict = {}
    for k, v in vocab.items():
        try:
            vocab_dict[str(k)] = v.decode('utf-8', errors='replace')
        except Exception as e:
            print(f'Warning: Could not decode token {k}: {e}')
            vocab_dict[str(k)] = repr(v)[2:-1]  # Remove b' and '
    
    with open('data/tokenizer/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    # Save merges
    print(f'Saving {len(merges)} merge rules...')
    with open('data/tokenizer/merges.txt', 'w', encoding='utf-8') as f:
        for t1, t2 in merges:
            try:
                line = f'{t1.decode("utf-8", errors="replace")} {t2.decode("utf-8", errors="replace")}\n'
                f.write(line)
            except Exception as e:
                print(f'Warning: Could not decode merge ({t1}, {t2}): {e}')
                continue
    
    print(f'SUCCESS: Generated BPE tokenizer!')
    print(f'Vocabulary size: {len(vocab)}')
    print(f'Number of merges: {len(merges)}')
    print('Files saved to data/tokenizer/')

if __name__ == '__main__':
    main()