#!/bin/bash

# =============================================================================
# RedPajama 1T Data Preprocessing Script for Megatron-LM
# =============================================================================
# This script preprocesses the RedPajama 1T dataset into Megatron's binary format
# and creates a data blend configuration file for training.
#
# Usage:
#   ./preprocess_redpajama.sh [--workers NUM] [--dry-run]
#
# Options:
#   --workers NUM    Number of worker processes (default: 64)
#   --dry-run        Print commands without executing
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
DATASET_ROOT="/workspace/dataset/redpajama-1t"
OUTPUT_ROOT="/workspace/processed_data/redpajama"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-8B"
WORKERS=${WORKERS:-64}
PARTITIONS=${PARTITIONS:-16}
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --partitions)
            PARTITIONS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --tokenizer-model)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Megatron root directory
MEGATRON_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Data blend output file
DATA_BLEND_FILE="${OUTPUT_ROOT}/data_blend.txt"

# Log file
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# Data Source Definitions
# Each source: NAME INPUT_PATTERN WEIGHT
# Weights are approximate proportions based on RedPajama data distribution
# =============================================================================
declare -A DATA_SOURCES=(
    ["arxiv"]="${DATASET_ROOT}/arxiv/*.jsonl"
    ["c4"]="${DATASET_ROOT}/c4/*.jsonl"
    ["github"]="${DATASET_ROOT}/github/*.jsonl"
    ["stackexchange"]="${DATASET_ROOT}/stackexchange/*.jsonl"
    ["wikipedia"]="${DATASET_ROOT}/wikipedia/*.jsonl"
    ["common_crawl_2019-30"]="${DATASET_ROOT}/common_crawl/2019-30/*.zst"
    ["common_crawl_2020-05"]="${DATASET_ROOT}/common_crawl/2020-05/*.zst"
    ["common_crawl_2021-04"]="${DATASET_ROOT}/common_crawl/2021-04/*.zst"
    ["common_crawl_2022-05"]="${DATASET_ROOT}/common_crawl/2022-05/*.zst"
    ["common_crawl_2023-06"]="${DATASET_ROOT}/common_crawl/2023-06/*.zst"
)

# Weights based on approximate token distribution in RedPajama
# Total should sum to 1.0
declare -A DATA_WEIGHTS=(
    ["arxiv"]="0.025"
    ["c4"]="0.15"
    ["github"]="0.045"
    ["stackexchange"]="0.02"
    ["wikipedia"]="0.02"
    ["common_crawl_2019-30"]="0.15"
    ["common_crawl_2020-05"]="0.15"
    ["common_crawl_2021-04"]="0.15"
    ["common_crawl_2022-05"]="0.15"
    ["common_crawl_2023-06"]="0.14"
)

# =============================================================================
# Helper Functions
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        log "Running: $*"
        eval "$@"
    fi
}

count_files() {
    local pattern="$1"
    local count=$(ls -1 $pattern 2>/dev/null | wc -l)
    echo "$count"
}

# =============================================================================
# Main Processing Function
# =============================================================================
process_source() {
    local name="$1"
    local input_pattern="$2"
    local output_prefix="${OUTPUT_ROOT}/${name}"
    local log_file="${LOG_DIR}/${name}.log"

    # Check if already processed
    if [ -f "${output_prefix}_text_document.bin" ] && [ -f "${output_prefix}_text_document.idx" ]; then
        log "Skipping ${name}: already processed"
        return 0
    fi

    # Count input files
    local file_count=$(count_files "$input_pattern")
    if [ "$file_count" -eq 0 ]; then
        log "WARNING: No files found for ${name} with pattern: ${input_pattern}"
        return 1
    fi

    log "Processing ${name}: ${file_count} files"

    # Determine partitions (use fewer partitions for small datasets)
    local use_partitions=$PARTITIONS
    if [ "$file_count" -lt "$PARTITIONS" ]; then
        use_partitions=$file_count
    fi

    # Ensure workers is divisible by partitions
    local use_workers=$WORKERS
    while [ $((use_workers % use_partitions)) -ne 0 ] && [ $use_partitions -gt 1 ]; do
        use_partitions=$((use_partitions - 1))
    done

    local cmd="python ${MEGATRON_ROOT}/tools/preprocess_data.py \
        --input '${input_pattern}' \
        --output-prefix '${output_prefix}' \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --workers ${use_workers} \
        --partitions ${use_partitions} \
        --append-eod \
        --json-keys text"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $cmd"
    else
        log "Command: $cmd"
        eval "$cmd" 2>&1 | tee "$log_file"
        local status=$?
        if [ $status -ne 0 ]; then
            log "ERROR: Failed to process ${name}"
            return $status
        fi
    fi

    return 0
}

# =============================================================================
# Generate Data Blend File
# =============================================================================
generate_blend_file() {
    log "Generating data blend file: ${DATA_BLEND_FILE}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would generate blend file with:"
        for name in "${!DATA_SOURCES[@]}"; do
            local weight="${DATA_WEIGHTS[$name]}"
            local prefix="${OUTPUT_ROOT}/${name}_text_document"
            echo "  ${weight} ${prefix}"
        done
        return 0
    fi

    # Create blend file
    > "$DATA_BLEND_FILE"

    for name in "${!DATA_SOURCES[@]}"; do
        local weight="${DATA_WEIGHTS[$name]}"
        local prefix="${OUTPUT_ROOT}/${name}_text_document"

        # Only add if files exist
        if [ -f "${prefix}.bin" ] && [ -f "${prefix}.idx" ]; then
            echo "${weight} ${prefix}" >> "$DATA_BLEND_FILE"
            log "Added to blend: ${weight} ${name}"
        else
            log "WARNING: Skipping ${name} in blend file (files not found)"
        fi
    done

    log "Data blend file created: ${DATA_BLEND_FILE}"
    echo ""
    echo "=== Data Blend File Contents ==="
    cat "$DATA_BLEND_FILE"
    echo "================================"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    log "=============================================="
    log "RedPajama 1T Preprocessing Script"
    log "=============================================="
    log "Dataset root: ${DATASET_ROOT}"
    log "Output root: ${OUTPUT_ROOT}"
    log "Tokenizer: ${TOKENIZER_TYPE} (${TOKENIZER_MODEL})"
    log "Workers: ${WORKERS}"
    log "Partitions: ${PARTITIONS}"
    log "Dry run: ${DRY_RUN}"
    log "=============================================="

    # Create output directory
    mkdir -p "$OUTPUT_ROOT"
    mkdir -p "$LOG_DIR"

    # Check for zstandard library
    if ! python -c "import zstandard" 2>/dev/null; then
        log "Installing zstandard library for .zst file support..."
        pip install zstandard
    fi

    # Process each data source
    local failed_sources=()
    for name in "${!DATA_SOURCES[@]}"; do
        local pattern="${DATA_SOURCES[$name]}"
        if ! process_source "$name" "$pattern"; then
            failed_sources+=("$name")
        fi
    done

    # Generate blend file
    generate_blend_file

    # Summary
    log "=============================================="
    log "Processing Complete"
    log "=============================================="

    if [ ${#failed_sources[@]} -gt 0 ]; then
        log "WARNING: The following sources failed to process:"
        for src in "${failed_sources[@]}"; do
            log "  - $src"
        done
    fi

    log ""
    log "To use in training, update your training script with:"
    log "  --data-args-path ${DATA_BLEND_FILE}"
    log ""
    log "Or for a single dataset, use:"
    log "  --data-path ${OUTPUT_ROOT}/<source>_text_document"
}

# Run main
main "$@"