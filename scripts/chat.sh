#!/usr/bin/env bash
# echo $#
if [ $# -eq 0 ]; then
    uv run python scripts/model_inference_cli.py \
    --interactive   
else
    uv run python scripts/model_inference_cli.py \
    --model "$1" \
    --interactive  
fi