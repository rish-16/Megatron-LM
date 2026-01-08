#!/bin/bash

# =============================================================================
# LLaMA 8B Training Script with RedPajama TINY Dataset (Development/Testing)
# =============================================================================
# This script is for quick development iteration using a small dataset.
# =============================================================================

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

EXP_NAME="llama3_8b_fp16_tiny_dev"
CHECKPOINT_PATH="checkpoints/$EXP_NAME"
WANDB_LOGS_PATH="wandb_logs/$EXP_NAME"
TOKENIZER_ARG="Qwen/Qwen3-8B"

# =============================================================================
# DATA CONFIGURATION - Using TINY dataset for development
# =============================================================================
DATA_PATH="/workspace/dataset/processed_data/redpajama_tiny/tiny_text_document"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$WANDB_LOGS_PATH"

# Distributed training setup
GPUS_PER_NODE=8
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

DTYPE="fp16"  # Mixed precision (bf16 compute, fp16 compatible)

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

# Reduced batch sizes for tiny dataset testing
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8  # Smaller for dev testing
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 10000  # Small number for testing
    --lr-decay-samples 9000
    --lr-warmup-samples 500
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

# Data arguments - using single tiny dataset
DATA_ARGS_LIST=(
    "--data-path $DATA_PATH"
    "--tokenizer-type HuggingFaceTokenizer"
    "--tokenizer-model $TOKENIZER_ARG"
    "--data-cache-path ${DATA_CACHE_PATH}"
    "--split '99,1,0'"
    "--no-create-attention-mask-in-dataloader"
    "--no-mmap-bin-files"
    "--num-workers 2"
    "--vocab-size 151936"
)

# Reduced logging for dev - more frequent to see progress
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 5
    --eval-interval 50
    --save-interval 500
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --wandb-project "hyper-project"
    --wandb-entity "asfeng"
    --wandb-exp-name "$EXP_NAME"
    --wandb-save-dir "$WANDB_LOGS_PATH"
)

# Run the training command
echo "=============================================="
echo "Running LLaMA 8B with TINY dataset (dev mode)"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Experiment: $EXP_NAME"
echo "=============================================="
echo ""
echo "Command:"
echo "torchrun ${DISTRIBUTED_ARGS[@]} \\"
echo "    $PRETRAIN_SCRIPT_PATH \\"
echo "    ${MODEL_ARGS[@]} \\"
echo "    ${TRAINING_ARGS[@]} \\"
echo "    ${DTYPE_ARGS[@]} \\"
echo "    ${MODEL_PARALLEL_ARGS[@]} \\"
echo "    ${DATA_ARGS_LIST[@]} \\"
echo "    ${EVAL_AND_LOGGING_ARGS[@]}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}