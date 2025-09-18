from .module.transformer import TransformerConfig


def model_params(config: TransformerConfig) -> int:
    """
    Get the total number of trainable model parameters.
    
    Args:
        config: TransformerConfig with model configuration
        
    Returns:
        int: Total number of trainable parameters
    """
    # Embedding is a vocab_size * d_model matrix lookup
    emb = config.vocab_size * config.d_model
    # RmsNorm has a (d_model,) 1D tensor as weights
    norm = config.d_model
    # (Multi-head) Self attention has 4 linear layers, each of size (d_model, d_model), for QKV and Output projection O.
    attention = 4 * config.d_model * config.d_model
    # Feed forward / SwiGLU has 3 linear layers / matrix multiplication, each of size (d_model, d_ff)
    swiglu = 3 * config.d_model * config.d_ff
    # Total # of params for one transformer block
    transformer_block = 2 * norm + attention + swiglu
    # Output linear layer is of size (d_model, vocab_size)
    out_linear = config.d_model * config.vocab_size
    print(f"Attention has {attention * config.num_layers:_} params, Feed forward {swiglu * config.num_layers:_} params, out linear {out_linear:_} params, emb {emb:_} params")
    return emb + config.num_layers * transformer_block + norm + out_linear


def flops(config: TransformerConfig) -> float:
    """
    Get total number of FLOPs to run forward pass on a single batch of input with (context_length, d_model) shape.
    
    Args:
        config: TransformerConfig with model configuration
        
    Returns:
        float: Total number of FLOPs for forward pass
    """
    # (Multi-head) Self attention has 4 linear layers, each of size (d_model, d_model), for QKV and Output projection O.
    # It takes input of shape (context_length, d_model) and produces output of shape (context_length, d_model)
    attention = (2 * config.context_length * (config.d_model ** 2)) * 4
    # Feed forward / SwiGLU has 3 linear layers / matrix multiplication, each of size (d_model, d_ff)
    # It takes input of shape (context_length, d_model) and produces output of shape (context_length, d_model)
    swiglu = (2 * config.context_length * config.d_model * config.d_ff) * 3
    # Output linear layer is of size (d_model, vocab_size)
    out_linear = (2 * config.d_model * config.vocab_size)

    total = (attention + swiglu) * config.num_layers + out_linear
    print(f"Attention consumes {attention * config.num_layers / total:.2%} flops, Feed forward {swiglu * config.num_layers / total:.2%} flops, and out linear {out_linear / total :.2%} flops")
    return total


def print_resource_summary(config: TransformerConfig, model_name: str = None) -> None:
    """
    Print a summary of model parameters and FLOPs.
    
    Args:
        config: TransformerConfig with model configuration
        model_name: Optional name for the model to include in output
    """
    if model_name:
        print(f"=== Resource Accounting for {model_name} ===")
    else:
        print("=== Resource Accounting ===")
    
    params = model_params(config)
    total_flops = flops(config)
    
    print(f"Total model params: {params:_}")
    print(f"Total FLOPs: {total_flops:_}")
    print()