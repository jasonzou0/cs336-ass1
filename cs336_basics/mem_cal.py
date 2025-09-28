def calc_memory(d_model, layers, batch_size=4, context=1024, vocab=50257):
    # Parameter calculation
    embed = vocab * d_model
    layer_params = (4 * d_model**2) + (8 * d_model**2) + (2 * d_model)
    transformer = layers * layer_params
    output = vocab * d_model
    total_params = embed + transformer + output
    
    # Memory calculation (in bytes)
    model_mem = total_params * 4
    grad_mem = total_params * 4
    opt_mem = total_params * 8
    act_mem = batch_size * context * d_model * layers * 4
    base = model_mem + grad_mem + opt_mem + act_mem
    total = base * 1.2  # 20% overhead
    
    return total_params, total / 1e9

configs = [
    (1280, 12, 'Original Large-ish'),
    (1280, 10, 'Reduce layers to 10'),
    (1152, 12, 'Reduce d_model to 1152'), 
    (1024, 14, 'Increase layers, reduce d_model'),
    (1200, 11, 'Balanced reduction')
]

print('Configuration Analysis for ~8GB target:')
print('='*60)
for d_model, layers, desc in configs:
    params, memory = calc_memory(d_model, layers)
    heads = d_model // 64
    print(f'{desc}:')
    print(f'  d_model={d_model}, layers={layers}, heads={heads}')
    print(f'  Parameters: {params/1e6:.1f}M, Memory: {memory:.2f}GB')
    print()