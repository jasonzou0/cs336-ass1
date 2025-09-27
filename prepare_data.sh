#!/bin/bash

# Data preparation script for CS336 training
# This script helps you prepare training data from the available text files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Prepare training data by tokenizing text files

OPTIONS:
    --dataset DATASET         Dataset to use (tinystories, owt, custom)
    --vocab-size SIZE         Vocabulary size (default: 50000)
    --method METHOD           Tokenization method (bpe, whitespace) (default: whitespace)
    --custom-train PATH       Path to custom training text file
    --custom-val PATH         Path to custom validation text file
    --help                    Show this help message

DATASETS:
    tinystories              Use TinyStoriesV2-GPT4 dataset
    owt                      Use OpenWebText dataset (large!)
    custom                   Use custom text files

EXAMPLES:
    # Prepare TinyStories dataset with whitespace tokenization
    $0 --dataset tinystories

    # Prepare TinyStories with BPE tokenization
    $0 --dataset tinystories --method bpe --vocab-size 32000

    # Prepare OpenWebText dataset (warning: very large!)
    $0 --dataset owt --vocab-size 50000

    # Use custom text files
    $0 --dataset custom --custom-train my_train.txt --custom-val my_val.txt

EOF
}

# Default values
DATASET=""
VOCAB_SIZE=50000
METHOD="whitespace"
CUSTOM_TRAIN=""
CUSTOM_VAL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --custom-train)
            CUSTOM_TRAIN="$2"
            shift 2
            ;;
        --custom-val)
            CUSTOM_VAL="$2"
            shift 2
            ;;
        --help|-h)
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

# Validate dataset argument
if [[ -z "$DATASET" ]]; then
    print_error "Dataset must be specified with --dataset"
    show_usage
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Set up file paths based on dataset
case $DATASET in
    tinystories)
        TRAIN_TEXT="$DATA_DIR/TinyStoriesV2-GPT4-train.txt"
        VAL_TEXT="$DATA_DIR/TinyStoriesV2-GPT4-valid.txt"
        TRAIN_BIN="$DATA_DIR/tinystories_train.bin"
        VAL_BIN="$DATA_DIR/tinystories_val.bin"
        ;;
    owt)
        TRAIN_TEXT="$DATA_DIR/owt_train.txt"
        VAL_TEXT="$DATA_DIR/owt_valid.txt"
        TRAIN_BIN="$DATA_DIR/owt_train.bin"
        VAL_BIN="$DATA_DIR/owt_val.bin"
        ;;
    custom)
        if [[ -z "$CUSTOM_TRAIN" ]]; then
            print_error "Custom training file must be specified with --custom-train"
            exit 1
        fi
        TRAIN_TEXT="$CUSTOM_TRAIN"
        VAL_TEXT="$CUSTOM_VAL"
        TRAIN_BIN="$DATA_DIR/custom_train.bin"
        VAL_BIN="$DATA_DIR/custom_val.bin"
        ;;
    *)
        print_error "Unknown dataset: $DATASET"
        print_info "Available datasets: tinystories, owt, custom"
        exit 1
        ;;
esac

# Check if text files exist
if [[ ! -f "$TRAIN_TEXT" ]]; then
    print_error "Training text file not found: $TRAIN_TEXT"
    exit 1
fi

if [[ -n "$VAL_TEXT" && ! -f "$VAL_TEXT" ]]; then
    print_warning "Validation text file not found: $VAL_TEXT"
    print_info "Will only prepare training data"
    VAL_TEXT=""
    VAL_BIN=""
fi

# Show configuration
print_info "Data Preparation Configuration:"
echo "  Dataset: $DATASET"
echo "  Vocabulary Size: $VOCAB_SIZE"
echo "  Tokenization Method: $METHOD"
echo "  Training Text: $TRAIN_TEXT"
if [[ -n "$VAL_TEXT" ]]; then
    echo "  Validation Text: $VAL_TEXT"
fi
echo "  Output Training Binary: $TRAIN_BIN"
if [[ -n "$VAL_BIN" ]]; then
    echo "  Output Validation Binary: $VAL_BIN"
fi
echo

# Show file sizes
print_info "Input File Sizes:"
echo "  Training: $(du -h "$TRAIN_TEXT" | cut -f1)"
if [[ -n "$VAL_TEXT" ]]; then
    echo "  Validation: $(du -h "$VAL_TEXT" | cut -f1)"
fi
echo

# Warning for large files
TRAIN_SIZE=$(stat -c%s "$TRAIN_TEXT")
if [[ $TRAIN_SIZE -gt 1000000000 ]]; then  # > 1GB
    print_warning "Large training file detected ($(du -h "$TRAIN_TEXT" | cut -f1))"
    print_warning "This may take a long time and use significant memory"
    echo -n "Continue? [y/N] "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Aborted by user"
        exit 0
    fi
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Prepare training data
print_info "Preparing training data..."
python prepare_data.py \
    --input "$TRAIN_TEXT" \
    --output "$TRAIN_BIN" \
    --vocab-size "$VOCAB_SIZE" \
    --method "$METHOD"

if [[ $? -ne 0 ]]; then
    print_error "Failed to prepare training data"
    exit 1
fi

# Prepare validation data if available
if [[ -n "$VAL_TEXT" && -n "$VAL_BIN" ]]; then
    print_info "Preparing validation data..."
    
    # For BPE, reuse the tokenizer trained on training data
    if [[ "$METHOD" == "bpe" ]]; then
        TOKENIZER_DIR="$DATA_DIR/tokenizer"
        if [[ -d "$TOKENIZER_DIR" ]]; then
            python prepare_data.py \
                --input "$VAL_TEXT" \
                --output "$VAL_BIN" \
                --vocab-size "$VOCAB_SIZE" \
                --method "$METHOD" \
                --reuse-tokenizer "$TOKENIZER_DIR"
        else
            print_warning "Tokenizer directory not found, training new tokenizer for validation data"
            python prepare_data.py \
                --input "$VAL_TEXT" \
                --output "$VAL_BIN" \
                --vocab-size "$VOCAB_SIZE" \
                --method "$METHOD"
        fi
    else
        python prepare_data.py \
            --input "$VAL_TEXT" \
            --output "$VAL_BIN" \
            --vocab-size "$VOCAB_SIZE" \
            --method "$METHOD"
    fi
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to prepare validation data"
        exit 1
    fi
fi

# Create convenient symlinks
print_info "Creating convenient symlinks..."
ln -sf "$(basename "$TRAIN_BIN")" "$DATA_DIR/train.bin"
if [[ -n "$VAL_BIN" ]]; then
    ln -sf "$(basename "$VAL_BIN")" "$DATA_DIR/val.bin"
fi

# Show results
print_success "Data preparation completed!"
echo
print_info "Generated Files:"
echo "  Training: $TRAIN_BIN ($(du -h "$TRAIN_BIN" | cut -f1))"
if [[ -n "$VAL_BIN" && -f "$VAL_BIN" ]]; then
    echo "  Validation: $VAL_BIN ($(du -h "$VAL_BIN" | cut -f1))"
fi
echo "  Symlinks: data/train.bin, data/val.bin"
echo

print_info "You can now run training with:"
echo "  ./train.sh --data-path data/train.bin --val-data-path data/val.bin --vocab-size $VOCAB_SIZE"
echo
print_info "Or for a quick test:"
echo "  ./train.sh --data-path data/train.bin --batch-size 8 --max-iters 1000 --d-model 256 --num-layers 4"