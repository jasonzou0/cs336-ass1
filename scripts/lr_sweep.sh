#!/bin/bash

# Array of learning rates to test
learning_rates=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2)

echo "Starting learning rate sweep with ${#learning_rates[@]} different values..."

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
    echo "========================================="
    echo "Running training with learning rate: $lr"
    echo "========================================="

    uv run scripts/trainer_cli.py \
        --train_data tinystories/TinyStoriesV2-GPT4-train_tokens.npy \
        --tokenizer_dir tinystories \
        --device=mps \
        --iterations=5000 \
        --batch_size 32 \
        --eval_data=tinystories/TinyStoriesV2-GPT4-valid_tokens.npy \
        --log_to_wandb \
        --learning_rate $lr

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Training with lr=$lr completed successfully"
    else
        echo "Training with lr=$lr failed"
        exit 1
    fi

    echo ""
done

echo "Learning rate sweep completed!"
