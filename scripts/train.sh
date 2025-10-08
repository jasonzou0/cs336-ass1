#!/usr/bin/env bash
# echo $#
############################################################
# Total		context_len	batchsize	iterations
# 327,680,000	256			16			80,000
# 327,680,000	256			32			40,000
# 327,680,000	256			64			20,000
# 327,680,000	256			128 		10,000
#
#  40,960,000	256			16  		10,000 
#  40,960,000	256			32  		 5,000 
#  40,960,000	256			64  		 2,500 
############################################################
parm='--d_model=512
    --d_ff=1344
    --n_layers=4
    --n_heads=16
    --batch_size=32
    --context_len=256
    --vocab_size=10000
    --theta=10000.0
    --compile
    --learning_rate=1e-4
    --lr_schedule_max_lr=1e-3
    --lr_schedule_min_lr=1e-6
    --lr_schedule_warmup_iters=150
    --lr_schedule_total_iters=3000
    --eval_every=100
    --eval_iters=10
    --log_every=1
    --save_every=100
    --max_iters=10000
    --min_loss=0.8'
if [ $# -eq 0 ]; then
    uv run python scripts/train_model_cli.py $parm
else
    uv run python scripts/train_model_cli.py $parm --ckpt="$1"
fi



# # --data
# # --training_data
# # --eval_data
# # tokenizer directory
# # --tokenizer_data
# # output directory
# # --output
# # data hyperparameters
# --batch_size
# --context_length
# --vocab_size
# # model hyperparameters
# --n_layers
# --n_heads
# --d_model
# --d_ff
# # learning rate scheduler hyperparameters
# --learning_rate
# --lr_schedule_max_lr
# --lr_schedule_min_lr
# --lr_schedule_warmup_iters
# --lr_schedule_total_iters
# # optimization hyperparameters
# --max_grad_norm
# --seed
# --device
# --dtype
# # training control
# --max_iters
# --log_every
# --min_loss
# --eval_every
# --eval_iters
# # checkpoint save/load
# --save_every
# --ckpt