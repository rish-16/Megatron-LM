#!/bin/bash

# =============================================================================
# Qwen3 3B Training Script with RedPajama TINY Dataset (Development/Testing)
# =============================================================================
# This script is for quick development iteration using a small dataset.
# Model: Qwen3-3B (3 billion parameters)
# =============================================================================

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

EXP_NAME="qwen3_3b_bf16_tiny_dev"
CHECKPOINT_PATH="checkpoints/$EXP_NAME"
WANDB_LOGS_PATH="wandb_logs/$EXP_NAME"
TOKENIZER_ARG="Qwen/Qwen3-4B"

# =============================================================================
# DATA CONFIGURATION - Using TINY dataset for development
# =============================================================================
DATA_PATH="/workspace/dataset/processed_data/redpajama_tiny/tiny_text_document"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$WANDB_LOGS_PATH"

# Distributed training setup - 8x 40GB GPUs on single node
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

PRETRAIN_SCRIPT_PATH="w_pretrain_gpt.py"

# Model parallelism configuration for 3B model on 8x 40GB GPUs
# Using TP=2 to reduce memory per GPU (especially for large vocab cross-entropy)
# The large vocab (151936) causes OOM during loss computation with TP=1
TP_SIZE=2   # Tensor parallelism to split vocab/attention across 2 GPUs
CP_SIZE=1   # No context parallelism needed
PP_SIZE=1   # No pipeline parallelism needed
# Effective DP = WORLD_SIZE / (TP * CP * PP) = 8 / 2 = 4

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

# =============================================================================
# Qwen3-3B Model Architecture
# Based on Qwen3 configuration:
# - hidden_size: 2048
# - num_hidden_layers: 36
# - num_attention_heads: 16
# - num_key_value_heads: 4 (GQA)
# - intermediate_size: 11008
# - vocab_size: 151936
# - max_position_embeddings: 40960
# - rope_theta: 1000000
# =============================================================================
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=40960
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 36
    --hidden-size 2048
    --ffn-hidden-size 11008
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.02
    --attention-backend fused
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-position-embedding
)

# Training hyperparameters for 8x 40GB GPUs with TP=2
# With TP=2, effective DP=4, so adjust global batch size
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16  # DP=4 * micro_batch=1 * gradient_accum=4
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 10000  # Small number for testing
    --lr-decay-samples 9000
    --lr-warmup-samples 500
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --initial-loss-scale 65536
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

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

# Logging and evaluation
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 5
    --eval-interval 50
    --save-interval 50
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
echo "Running Qwen3 3B with TINY dataset (dev mode)"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Experiment: $EXP_NAME"
echo "Model: Qwen3-3B (36 layers, hidden=2048, heads=16, GQA groups=4)"
echo "=============================================="
echo ""
echo "Command:"
echo "torchrun ${DISTRIBUTED_ARGS[@]} \\"
echo "    $PRETRAIN_SCRIPT_PATH \\"
echo "    ${MODEL_ARGS[@]} \\"
echo "    ${TRAINING_ARGS[@]} \\"
echo "    ${MODEL_PARALLEL_ARGS[@]} \\"
echo "    ${DATA_ARGS_LIST[@]} \\"
echo "    ${EVAL_AND_LOGGING_ARGS[@]}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}