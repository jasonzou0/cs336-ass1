    <Module test_data.py
      test_get_batch

    <Module test_model.py
      test_model.py::test_linear
      test_model.py::test_embedding
      test_model.py::test_swiglu
      test_model.py::test_scaled_dot_product_attention
      test_model.py::test_4d_scaled_dot_product_attention
      test_model.py::test_multihead_self_attention
      test_model.py::test_multihead_self_attention_with_rope
      test_model.py::test_transformer_lm
      test_model.py::test_transformer_lm_truncated_input
      test_model.py::test_transformer_block
      test_model.py::test_rmsnorm
      test_model.py::test_rope
      test_model.py::test_silu_matches_pytorch
      
    <Module test_nn_utils.py
      test_softmax_matches_pytorch
      test_cross_entropy
      test_gradient_clipping

    <Module test_optimizer.py
      test_optimizer.py::test_adamw
      test_optimizer.py::test_get_lr_cosine_schedule

    <Module test_serialization.py
      test_serialization.py::test_checkpointing

    <Module test_tokenizer.py::
      test_tokenizer.py::test_roundtrip_empty
      test_tokenizer.py::test_empty_matches_tiktoken
      test_tokenizer.py::test_roundtrip_single_character
      test_tokenizer.py::test_single_character_matches_tiktoken
      test_tokenizer.py::test_roundtrip_single_unicode_character
      test_tokenizer.py::test_single_unicode_character_matches_tiktoken
      test_tokenizer.py::test_roundtrip_ascii_string
      test_tokenizer.py::test_ascii_string_matches_tiktoken
      test_tokenizer.py::test_roundtrip_unicode_string
      test_tokenizer.py::test_unicode_string_matches_tiktoken
      test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens
      test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken
      test_tokenizer.py::test_overlapping_special_tokens
      test_tokenizer.py::test_address_roundtrip
      test_tokenizer.py::test_address_matches_tiktoken
      test_tokenizer.py::test_german_roundtrip
      test_tokenizer.py::test_german_matches_tiktoken
      test_tokenizer.py::test_tinystories_sample_roundtrip
      test_tokenizer.py::test_tinystories_matches_tiktoken
      test_tokenizer.py::test_encode_special_token_trailing_newlines
      test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace
      test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip
      test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken
      test_tokenizer.py::test_encode_iterable_memory_usage
      test_tokenizer.py::test_encode_memory_usage
    

      test_train_bpe.py::test_train_bpe_speed
      test_train_bpe.py::test_train_bpe
      test_train_bpe.py::test_train_bpe_special_tokens

