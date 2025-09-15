"""
Unit tests for token validation utilities.
"""

import unittest
import tempfile
import os
import numpy as np

from cs336_basics.token_validation import validate_special_tokens_in_data
from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


class TestTokenValidation(unittest.TestCase):
    
    def setUp(self):
        # Sample text with special tokens
        self.sample_text = [
            "Hello world! This is a test.",
            "<|endoftext|>", 
            "Another sentence with some text.",
            "More text here <|endoftext|> and after."
        ]
        
        # Create temp directory and train BPE once for both tests
        self.temp_dir = tempfile.mkdtemp()
        text_file = os.path.join(self.temp_dir, "train.txt")
        with open(text_file, 'w') as f:
            f.write('\n'.join(self.sample_text))
        
        self.vocab, self.merges = train_bpe(text_file, 1000, special_tokens=["<|endoftext|>"])
        
        self.addCleanup(self._cleanup)
    
    def _cleanup(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_good_case(self):
        """Test validation passes when special tokens are correctly handled."""
        # Tokenizer WITH special tokens (correct)
        tokenizer = Tokenizer(self.vocab, self.merges, special_tokens=["<|endoftext|>"])
        
        tokens = []
        for text in self.sample_text:
            tokens.extend(tokenizer.encode(text))
        tokens = np.array(tokens)
        
        # Should NOT find bad tokenizations
        result = validate_special_tokens_in_data(tokens, self.vocab, self.merges, ["<|endoftext|>"])
        self.assertTrue(result)
    
    def test_bad_case(self):
        """Test validation fails when special tokens are incorrectly handled."""
        # Tokenizer WITHOUT special tokens (incorrect - simulates the bug)
        tokenizer = Tokenizer(self.vocab, self.merges, special_tokens=[])
        
        tokens = []
        for text in self.sample_text:
            tokens.extend(tokenizer.encode(text))
        tokens = np.array(tokens)
        
        # SHOULD find bad tokenizations
        result = validate_special_tokens_in_data(tokens, self.vocab, self.merges, ["<|endoftext|>"])
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()