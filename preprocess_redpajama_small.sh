#!/bin/bash

# =============================================================================
# RedPajama SMALL Dataset Preprocessing Script
# =============================================================================
# Creates a small test dataset for development/debugging purposes.
# Processes only wikipedia (1 file) and stackexchange (1 file) - fastest sources.
# =============================================================================

set -e

# Configuration
DATASET_ROOT="/workspace/dataset/redpajama-1t"
OUTPUT_ROOT="/workspace/processed_data/redpajama_small"
TOKENIZER_TYPE="HuggingFaceTokenizer"
TOKENIZER_MODEL="Qwen/Qwen3-8B"
WORKERS=${WORKERS:-16}

MEGATRON_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_BLEND_FILE="${OUTPUT_ROOT}/data_blend.txt"
LOG_DIR="${OUTPUT_ROOT}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"

log "=============================================="
log "RedPajama SMALL Dataset Preprocessing"
log "=============================================="
log "Output: ${OUTPUT_ROOT}"
log "Tokenizer: ${TOKENIZER_MODEL}"
log "Workers: ${WORKERS}"
log "=============================================="

# Process Wikipedia (single file, relatively small)
if [ ! -f "${OUTPUT_ROOT}/wikipedia_text_document.bin" ]; then
    log "Processing wikipedia..."
    python ${MEGATRON_ROOT}/tools/preprocess_data.py \
        --input "${DATASET_ROOT}/wikipedia/wiki.jsonl" \
        --output-prefix "${OUTPUT_ROOT}/wikipedia" \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --workers ${WORKERS} \
        --partitions 1 \
        --append-eod \
        --json-keys text \
        2>&1 | tee "${LOG_DIR}/wikipedia.log"
    log "Wikipedia processing complete!"
else
    log "Skipping wikipedia: already processed"
fi

# Process StackExchange (single file)
if [ ! -f "${OUTPUT_ROOT}/stackexchange_text_document.bin" ]; then
    log "Processing stackexchange..."
    python ${MEGATRON_ROOT}/tools/preprocess_data.py \
        --input "${DATASET_ROOT}/stackexchange/stackexchange.jsonl" \
        --output-prefix "${OUTPUT_ROOT}/stackexchange" \
        --tokenizer-type ${TOKENIZER_TYPE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --workers ${WORKERS} \
        --partitions 1 \
        --append-eod \
        --json-keys text \
        2>&1 | tee "${LOG_DIR}/stackexchange.log"
    log "StackExchange processing complete!"
else
    log "Skipping stackexchange: already processed"
fi

# Generate data blend file
log "Generating data blend file..."
cat > "$DATA_BLEND_FILE" << EOF
0.5 ${OUTPUT_ROOT}/wikipedia_text_document
0.5 ${OUTPUT_ROOT}/stackexchange_text_document
EOF

log "=============================================="
log "SMALL Dataset Processing Complete!"
log "=============================================="
log ""
log "Output files:"
ls -lh ${OUTPUT_ROOT}/*.bin ${OUTPUT_ROOT}/*.idx 2>/dev/null || echo "  (files still processing)"
log ""
log "Data blend file: ${DATA_BLEND_FILE}"
cat "$DATA_BLEND_FILE"
log ""
log "To use in training:"
log "  --data-args-path ${DATA_BLEND_FILE}"
log ""
log "Or use single dataset:"
log "  --data-path ${OUTPUT_ROOT}/wikipedia_text_document"