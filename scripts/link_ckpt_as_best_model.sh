#!/usr/bin/env bash
best_ckpt=$(realpath "$1")
rm ./output/inference_model.bin && ln -s "$best_ckpt" "./output/inference_model.bin"