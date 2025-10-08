#!/usr/bin/env bash
best_model="/DATA/Sync/Files/Programming/AI_ML/CS336/github/cs336-ass1/output/inference_model.bin"
best_ckpt=$(realpath "$1")
rm "$best_model" 
ln -s "$best_ckpt" "$best_model"