#!/bin/bash

# =============================================================================
# HELM-Qwen3 3B Training Script with RedPajama TINY Dataset (Development/Testing)
# =============================================================================
# This script trains a Hyperbolic Efficient Language Model (HELM) based on
# the Qwen3-3B architecture, replacing key components with hyperbolic versions:
# - Lorentz Multi-head Latent Attention (HMLA)
# - Lorentz Mixture of Curvature Experts (MiCE)
#
# Model: HELM-Qwen3-3B (~3 billion parameters with hyperbolic layers)
# =============================================================================

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

EXP_NAME="helm_qwen3_3b_bf16_tiny_dev"
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

PRETRAIN_SCRIPT_PATH="pretrain_helm_qwen3.py"

# Model parallelism configuration for HELM 3B model on 8x 40GB GPUs
# Using TP=2 to reduce memory per GPU (hyperbolic ops may use more memory)
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
# HELM-Qwen3-3B Model Architecture
# Based on Qwen3-3B with hyperbolic modifications:
# - hidden_size: 2048 (spatial dimension in Lorentz space is 2047)
# - num_hidden_layers: 36
# - num_attention_heads: 16
# - vocab_size: 151936
# - max_position_embeddings: 40960
# - rope_theta: 1000000
#
# Hyperbolic-specific:
# - MLA with Lorentz attention mechanism
# - MoE with multiple curvature experts
# =============================================================================
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=40960
MODEL_ARGS=(
    --num-layers 36
    --hidden-size 2048
    --ffn-hidden-size 11008
    --num-attention-heads 16
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.02
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --no-position-embedding
)

# =============================================================================
# HELM-specific Model Arguments
# =============================================================================
# MLA (Multi-head Latent Attention) parameters
MLA_ARGS=(
    --helm-q-lora-rank 0              # No LoRA for Q (full projection)
    --helm-kv-lora-rank 512           # LoRA rank for KV compression
    --helm-qk-nope-head-dim 128       # Non-positional QK dimension
    --helm-qk-rope-head-dim 64        # RoPE QK dimension
    --helm-v-head-dim 128             # Value head dimension
)

# MoE (Mixture of Curvature Experts) parameters
MOE_ARGS=(
    --mice-inter-dim 1408        # Expert intermediate dimension
    --n-routed-experts 8         # Number of routed experts
    --n-shared-experts 1         # Number of shared experts
    --n-activated-experts 2      # Top-k experts per token
    --n-expert-groups 1          # Expert groups
    --n-limited-groups 1         # Limited groups for routing
    --n-dense-layers 2           # Dense FFN layers before MoE
    --score-func softmax         # Expert routing score function
    --route-scale 1.0            # Routing scale factor
    --bias-update-speed 0.001    # Bias update speed for load balancing
)

# Hyperbolic parameters
HYPERBOLIC_ARGS=(
    --curvature 1.0              # Initial curvature of hyperbolic space
    --train-curvature            # Make curvature learnable
    --project-emb                # Project Euclidean embeddings to hyperbolic
    --helm-beta-fast 32          # RoPE scaling parameter
    --helm-beta-slow 1           # RoPE scaling parameter
    --helm-mscale 1.0            # Attention scaling
    --helm-rope-factor 1.0       # RoPE scaling factor
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
echo "Running HELM-Qwen3 3B with TINY dataset (dev mode)"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Experiment: $EXP_NAME"
echo "Model: HELM-Qwen3-3B (36 layers, hidden=2048, heads=16)"
echo "Hyperbolic: Lorentz MLA + MoE with 8 curvature experts"
echo "=============================================="
echo ""
echo "Command:"
echo "torchrun ${DISTRIBUTED_ARGS[@]} \\"
echo "    $PRETRAIN_SCRIPT_PATH \\"
echo "    ${MODEL_ARGS[@]} \\"
echo "    ${MLA_ARGS[@]} \\"
echo "    ${MOE_ARGS[@]} \\"
echo "    ${HYPERBOLIC_ARGS[@]} \\"
echo "    ${TRAINING_ARGS[@]} \\"
echo "    ${MODEL_PARALLEL_ARGS[@]} \\"
echo "    ${DATA_ARGS_LIST[@]} \\"
echo "    ${EVAL_AND_LOGGING_ARGS[@]}"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${HYPERBOLIC_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
