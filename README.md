# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
sh scripts/download_tiny_stories.sh
```

### Install packages and jupyter kernel

``` sh
sh scripts/build_kernel.sh
```

## Training

### BPE tokenizer 

Tokenizer Training:

```sh
uv run scripts/train_bpe_cli.py --input_path ${INPUT_DATA} --vocab_size ${VOCAB_SIZE} --output_dir ${OUTPUT_DIR}
```
Optionally add `--load_pretokenization` to load pre-existing pretokenization result instead of computing it from scratch.

Running tokenization on text file:

```sh
uv run python3 scripts/tokenizer_cli.py --artifact_dir=${BPE_ARTIFACT_DIR} --input_text=${INPUT_TEXT_FILE}  --output_directory=${OUTPUT_DIR}
```
where `${BPE_ARTIFACT_DIR}` contains the output merges and vocab files from tokenizer training.

### Transformer

Training:

```sh
uv run scripts/trainer_cli.py --train_data {TRAIN_TOKENS_DATA} --tokenizer_dir ${BPE_ARTIFACT_DIR}  --eval_data {EVAL_TOKENS_DATA} --device=mps  --num_batches=${TRAIN_BATCHES} --checkpoint_interval=${CHECKPOINT_INTERVAL}
```