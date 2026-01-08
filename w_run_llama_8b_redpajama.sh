#!/bin/bash

# =============================================================================
# LLaMA 8B Training Script with RedPajama 1T Dataset
# =============================================================================
# This script trains a LLaMA 8B model using the preprocessed RedPajama dataset.
# Run preprocess_redpajama.sh first to prepare the data.
# =============================================================================

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

EXP_NAME="llama3_8b_fp8_redpajama"
CHECKPOINT_PATH="checkpoints/$EXP_NAME"
WANDB_LOGS_PATH="wandb_logs/$EXP_NAME"
TOKENIZER_ARG="Qwen/Qwen3-8B"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Option 1: Use the data blend file (recommended for multi-source training)
DATA_BLEND_FILE="/workspace/processed_data/redpajama/data_blend.txt"

# Option 2: Use a single dataset (uncomment to use)
# SINGLE_DATA_PATH="/workspace/processed_data/redpajama/c4_text_document"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$WANDB_LOGS_PATH"

# Distributed training setup
GPUS_PER_NODE=1
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

PRETRAIN_SCRIPT_PATH="w_pretrain_gpt.py"

# Model parallelism
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1

DTYPE="fp8"

# Data cache path
DATA_CACHE_PATH="/workspace/checkpoints/data_cache_${EXP_NAME}"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=8192
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 1
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 1953125000
    --lr-decay-samples 1949218748
    --lr-warmup-samples 3906252
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4
    --decoupled-min-lr 4.5e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
    --exit-duration-in-mins 235
)

# FP8 arguments
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
)

# DDP arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# =============================================================================
# DATA ARGUMENTS - Choose one of the options below
# =============================================================================

# Option 1: Use data blend file (multi-source weighted training)
if [ -f "$DATA_BLEND_FILE" ]; then
    echo "Using data blend file: $DATA_BLEND_FILE"
    DATA_ARGS_LIST=(
        "--data-args-path $DATA_BLEND_FILE"
        "--tokenizer-type HuggingFaceTokenizer"
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 4"
        "--vocab-size 128256"
    )
# Option 2: Use single dataset
elif [ -n "$SINGLE_DATA_PATH" ] && [ -f "${SINGLE_DATA_PATH}.bin" ]; then
    echo "Using single dataset: $SINGLE_DATA_PATH"
    DATA_ARGS_LIST=(
        "--data-path $SINGLE_DATA_PATH"
        "--tokenizer-type HuggingFaceTokenizer"
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 4"
        "--vocab-size 128256"
    )
else
    echo "ERROR: No valid data configuration found!"
    echo "Please run preprocess_redpajama.sh first, or check your data paths."
    echo ""
    echo "Expected data blend file: $DATA_BLEND_FILE"
    exit 1
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --wandb-project "hyper-project"
    --wandb-entity "asfeng"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$WANDB_LOGS_PATH"
)

# Run the training command
echo "=============================================="
echo "Running command:"
echo "torchrun ${DISTRIBUTED_ARGS[@]} \\"
echo "    $PRETRAIN_SCRIPT_PATH \\"
echo "    ${MODEL_ARGS[@]} \\"
echo "    ${TRAINING_ARGS[@]} \\"
echo "    ${DTYPE_ARGS[@]} \\"
echo "    ${MODEL_PARALLEL_ARGS[@]} \\"
echo "    ${DATA_ARGS_LIST[@]} \\"
echo "    ${EVAL_AND_LOGGING_ARGS[@]}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}