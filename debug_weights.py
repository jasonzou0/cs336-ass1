#!/usr/bin/env python3

import sys
import os
sys.path.append('tests')

import torch
from tests.conftest import *

def main():
    # Get the fixture data
    in_embeddings = torch.randn(1, 4, 64)  # batch=1, seq=4, d_model=64
    d_model = 64
    n_heads = 4
    
    # Load state dict
    try:
        import tests.conftest as conftest
        
        # Try to get the fixture data
        ts_state_dict = None
        
        # Let me read the specific test
        import pytest
        from tests.test_model import test_multihead_self_attention
        
        # Load the ts fixtures
        fixture_dir = "tests/fixtures/ts_tests"
        if os.path.exists(fixture_dir):
            state_dict = torch.load(os.path.join(fixture_dir, "model.pt"), map_location='cpu')
            print("State dict keys:")
            for key in sorted(state_dict.keys()):
                if 'layers.0.attn' in key:
                    print(f"  {key}: {state_dict[key].shape}")
                    
        # Also check specific weights
        if 'layers.0.attn.q_proj.weight' in state_dict:
            q_weight = state_dict['layers.0.attn.q_proj.weight']
            k_weight = state_dict['layers.0.attn.k_proj.weight']
            v_weight = state_dict['layers.0.attn.v_proj.weight']
            o_weight = state_dict['layers.0.attn.output_proj.weight']
            
            print(f"\nWeight shapes:")
            print(f"Q: {q_weight.shape}")
            print(f"K: {k_weight.shape}")
            print(f"V: {v_weight.shape}")
            print(f"O: {o_weight.shape}")
            
            print(f"\nd_model={d_model}, n_heads={n_heads}, d_k={d_model//n_heads}")
            
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
