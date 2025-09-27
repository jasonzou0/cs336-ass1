#!/bin/bash

# CS336 Transformer Training Script
# This script provides easy configuration and execution of the transformer training

set -e  # Exit on any error

# Default configuration
DATA_PATH=""
VAL_DATA_PATH=""
OUT_DIR="./checkpoints"
CONFIG_FILE=""
RESUME_CHECKPOINT=""

# Model defaults
VOCAB_SIZE=50257
CONTEXT_LENGTH=1024
D_MODEL=768
NUM_LAYERS=12
NUM_HEADS=12
D_FF=3072
ROPE_THETA=10000.0

# Training defaults
BATCH_SIZE=32
MAX_ITERS=100000
LEARNING_RATE=1e-4
MIN_LEARNING_RATE=1e-5
WARMUP_ITERS=2000
WEIGHT_DECAY=1e-2
GRAD_CLIP=1.0

# Logging defaults
EVAL_INTERVAL=1000
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=5000
EVAL_ITERS=200

# System defaults
DEVICE="auto"
COMPILE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

CS336 Transformer Training Script

REQUIRED OPTIONS:
    --data-path PATH          Path to training data file (.bin format)

OPTIONAL OPTIONS:
    --val-data-path PATH      Path to validation data file
    --out-dir PATH            Output directory for checkpoints (default: ./checkpoints)
    --config FILE             Load configuration from JSON file
    --resume PATH             Resume training from checkpoint
    
MODEL CONFIGURATION:
    --vocab-size N            Vocabulary size (default: $VOCAB_SIZE)
    --context-length N        Context length (default: $CONTEXT_LENGTH)
    --d-model N               Model dimension (default: $D_MODEL)
    --num-layers N            Number of transformer layers (default: $NUM_LAYERS)
    --num-heads N             Number of attention heads (default: $NUM_HEADS)
    --d-ff N                  Feed-forward dimension (default: $D_FF)
    --rope-theta F            RoPE theta parameter (default: $ROPE_THETA)

TRAINING CONFIGURATION:
    --batch-size N            Batch size (default: $BATCH_SIZE)
    --max-iters N             Maximum training iterations (default: $MAX_ITERS)
    --learning-rate F         Initial learning rate (default: $LEARNING_RATE)
    --min-learning-rate F     Minimum learning rate (default: $MIN_LEARNING_RATE)
    --warmup-iters N          Warmup iterations (default: $WARMUP_ITERS)
    --weight-decay F          Weight decay (default: $WEIGHT_DECAY)
    --grad-clip F             Gradient clipping threshold (default: $GRAD_CLIP)

LOGGING CONFIGURATION:
    --eval-interval N         Evaluation interval (default: $EVAL_INTERVAL)
    --log-interval N          Logging interval (default: $LOG_INTERVAL)
    --checkpoint-interval N   Checkpoint saving interval (default: $CHECKPOINT_INTERVAL)
    --eval-iters N            Number of iterations for evaluation (default: $EVAL_ITERS)

SYSTEM CONFIGURATION:
    --device DEVICE           Device to use (auto, cuda, cpu) (default: $DEVICE)
    --compile                 Enable torch.compile optimization
    --dry-run                 Show command that would be executed without running it

EXAMPLES:
    # Basic training
    $0 --data-path data/train.bin --batch-size 16 --max-iters 50000

    # Training with validation
    $0 --data-path data/train.bin --val-data-path data/val.bin --out-dir my_checkpoints

    # Small model for testing
    $0 --data-path data/train.bin --d-model 256 --num-layers 6 --num-heads 8 --batch-size 16

    # Resume training
    $0 --config checkpoints/config.json --resume checkpoints/ckpt_010000.pt

    # Use configuration file
    $0 --config my_config.json

    # Dry run to see the command
    $0 --data-path data/train.bin --dry-run

EOF
}

# Function to create a sample configuration file
create_sample_config() {
    local config_file="sample_config.json"
    cat > "$config_file" << EOF
{
  "data_path": "data/train.bin",
  "val_data_path": "data/val.bin",
  "out_dir": "./checkpoints",
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
  "rope_theta": 10000.0,
  "batch_size": 32,
  "max_iters": 100000,
  "learning_rate": 1e-4,
  "min_learning_rate": 1e-5,
  "warmup_iters": 2000,
  "weight_decay": 1e-2,
  "grad_clip": 1.0,
  "eval_interval": 1000,
  "log_interval": 100,
  "checkpoint_interval": 5000,
  "eval_iters": 200,
  "device": "auto",
  "compile": false
}
EOF
    print_success "Created sample configuration file: $config_file"
}

# Parse command line arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --val-data-path)
            VAL_DATA_PATH="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --context-length)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --d-ff)
            D_FF="$2"
            shift 2
            ;;
        --rope-theta)
            ROPE_THETA="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --min-learning-rate)
            MIN_LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup-iters)
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --grad-clip)
            GRAD_CLIP="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        --eval-iters)
            EVAL_ITERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --compile)
            COMPILE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sample-config)
            create_sample_config
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$CONFIG_FILE" && -z "$DATA_PATH" ]]; then
    print_error "Either --data-path or --config must be provided"
    echo
    show_usage
    exit 1
fi

if [[ -n "$CONFIG_FILE" && ! -f "$CONFIG_FILE" ]]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [[ -n "$DATA_PATH" && ! -f "$DATA_PATH" ]]; then
    print_error "Training data file not found: $DATA_PATH"
    exit 1
fi

if [[ -n "$VAL_DATA_PATH" && ! -f "$VAL_DATA_PATH" ]]; then
    print_error "Validation data file not found: $VAL_DATA_PATH"
    exit 1
fi

if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
    print_error "Resume checkpoint file not found: $RESUME_CHECKPOINT"
    exit 1
fi

# Build the command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/cs336_basics/my_training.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Training script not found: $PYTHON_SCRIPT"
    exit 1
fi

CMD="python3 $PYTHON_SCRIPT"

# Add arguments
if [[ -n "$CONFIG_FILE" ]]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

if [[ -n "$DATA_PATH" ]]; then
    CMD="$CMD --data-path $DATA_PATH"
fi

if [[ -n "$VAL_DATA_PATH" ]]; then
    CMD="$CMD --val-data-path $VAL_DATA_PATH"
fi

if [[ -n "$OUT_DIR" ]]; then
    CMD="$CMD --out-dir $OUT_DIR"
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    CMD="$CMD --resume $RESUME_CHECKPOINT"
fi

# Add model parameters
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --context-length $CONTEXT_LENGTH"
CMD="$CMD --d-model $D_MODEL"
CMD="$CMD --num-layers $NUM_LAYERS"
CMD="$CMD --num-heads $NUM_HEADS"
CMD="$CMD --d-ff $D_FF"
CMD="$CMD --rope-theta $ROPE_THETA"

# Add training parameters
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --max-iters $MAX_ITERS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --min-learning-rate $MIN_LEARNING_RATE"
CMD="$CMD --warmup-iters $WARMUP_ITERS"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --grad-clip $GRAD_CLIP"

# Add logging parameters
CMD="$CMD --eval-interval $EVAL_INTERVAL"
CMD="$CMD --log-interval $LOG_INTERVAL"
CMD="$CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
CMD="$CMD --eval-iters $EVAL_ITERS"

# Add system parameters
CMD="$CMD --device $DEVICE"

if [[ "$COMPILE" == "true" ]]; then
    CMD="$CMD --compile"
fi

# Show configuration summary
print_info "Training Configuration:"
echo "  Data Path: ${DATA_PATH:-$CONFIG_FILE}"
echo "  Output Directory: $OUT_DIR"
echo "  Model: d_model=$D_MODEL, layers=$NUM_LAYERS, heads=$NUM_HEADS"
echo "  Training: batch_size=$BATCH_SIZE, max_iters=$MAX_ITERS, lr=$LEARNING_RATE"
echo "  Device: $DEVICE"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "  Resume from: $RESUME_CHECKPOINT"
fi
echo

# Create output directory
if [[ "$DRY_RUN" == "false" ]]; then
    mkdir -p "$OUT_DIR"
    print_info "Created output directory: $OUT_DIR"
fi

# Execute or show the command
if [[ "$DRY_RUN" == "true" ]]; then
    print_info "Dry run - would execute:"
    echo "$CMD"
else
    print_info "Starting training..."
    echo "Command: $CMD"
    echo
    
    # Change to script directory to ensure relative imports work
    cd "$SCRIPT_DIR"
    
    # Execute the command
    eval "$CMD"
    
    if [[ $? -eq 0 ]]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed!"
        exit 1
    fi
fi