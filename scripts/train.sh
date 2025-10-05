#!/usr/bin/env bash
# echo $#
if [ $# -eq 0 ]; then
    uv run python scripts/train_model_cli.py \
    --lr_schedule_total_iters=3000 \
    --log_every=1 \
    --vocab_size=10000 \
    --d_model=512 \
    --d_ff=1344 \
    --n_layers=4 \
    --n_heads=16 \
    --context_len=256 \
    --batch_size=16 \
    --save_every=50 \
    --eval_every=100 \
    --max_iters=3000 
else
    uv run python scripts/train_model_cli.py \
    --lr_schedule_total_iters=3000 \
    --log_every=1 \
    --vocab_size=10000 \
    --d_model=512 \
    --d_ff=1344 \
    --n_layers=4 \
    --n_heads=16 \
    --context_len=256 \
    --batch_size=16 \
    --save_every=50 \
    --eval_every=100 \
    --max_iters=3000 \
    --ckpt="$1"
fi