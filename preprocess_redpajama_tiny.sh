#!/bin/bash

# =============================================================================
# RedPajama TINY Dataset Preprocessing Script
# =============================================================================
# Creates a TINY test dataset (~10k samples) for fast development iteration.
# Extracts first N lines from wikipedia for quick testing.
# =============================================================================

set -e

# Configuration
DATASET_ROOT="/workspace/dataset/redpajama-1t"
OUTPUT_ROOT="/workspace/processed_data/redpajama_tiny"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-8B"
WORKERS=${WORKERS:-8}
NUM_SAMPLES=${NUM_SAMPLES:-10000}  # Number of documents to extract

MEGATRON_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEMP_FILE="${OUTPUT_ROOT}/tiny_subset.jsonl"
LOG_DIR="${OUTPUT_ROOT}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

log "=============================================="
log "RedPajama TINY Dataset Preprocessing"
log "=============================================="
log "Output: ${OUTPUT_ROOT}"
log "Samples: ${NUM_SAMPLES}"
log "Tokenizer: ${TOKENIZER_MODEL}"
log "Workers: ${WORKERS}"
log "=============================================="

# Create tiny subset from wikipedia
if [ ! -f "$TEMP_FILE" ]; then
    log "Extracting ${NUM_SAMPLES} samples from wikipedia..."
    head -n ${NUM_SAMPLES} "${DATASET_ROOT}/wikipedia/wiki.jsonl" > "$TEMP_FILE"
    log "Extracted $(wc -l < "$TEMP_FILE") samples"
else
    log "Tiny subset already exists: $TEMP_FILE"
fi

# Process the tiny subset
if [ ! -f "${OUTPUT_ROOT}/tiny_text_document.bin" ]; then
    log "Processing tiny dataset..."
    python ${MEGATRON_ROOT}/tools/preprocess_data.py \
        --input "$TEMP_FILE" \
        --output-prefix "${OUTPUT_ROOT}/tiny" \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --workers ${WORKERS} \
        --partitions 1 \
        --append-eod \
        --json-keys text \
        2>&1 | tee "${LOG_DIR}/tiny.log"
    log "Processing complete!"
else
    log "Skipping: already processed"
fi

# Generate data blend file (single source)
DATA_BLEND_FILE="${OUTPUT_ROOT}/data_blend.txt"
echo "1.0 ${OUTPUT_ROOT}/tiny_text_document" > "$DATA_BLEND_FILE"

log "=============================================="
log "TINY Dataset Processing Complete!"
log "=============================================="
log ""
log "Output files:"
ls -lh ${OUTPUT_ROOT}/*.bin ${OUTPUT_ROOT}/*.idx 2>/dev/null || echo "  (processing...)"
log ""
log "To use in training:"
log "  --data-path ${OUTPUT_ROOT}/tiny_text_document"
log ""
log "Or with blend file:"
log "  --data-args-path ${DATA_BLEND_FILE}"