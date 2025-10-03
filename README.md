# Build a LLM from scratch

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
uv run scripts/trainer_cli.py --train_data {TRAIN_TOKENS_DATA} --tokenizer_dir ${BPE_ARTIFACT_DIR}  --eval_data {EVAL_TOKENS_DATA} --device=mps  --iterations=${TRAIN_BATCHES} --checkpoint_interval=${CHECKPOINT_INTERVAL} --log_to_wandb
```

Decoding:

```sh
uv run scripts/decoder_cli.py --model ${MODEL_CHECKPOINT} --tokenizer_dir ${BPE_ARTIFACT_DIR} --context_length 256 --device mps --max_new_tokens 200
```