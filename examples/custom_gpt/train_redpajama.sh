#!/bin/bash

# Training script for hyperbolic transformer on RedPajama

# Data paths
DATA_PATH="/path/to/processed/redpajama"
CHECKPOINT_PATH="/path/to/checkpoints"
VOCAB_FILE="/path/to/vocab.json"
MERGE_FILE="/path/to/merges.txt"

# Model configuration
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Training hyperparameters
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128
TP_SIZE=4
PP_SIZE=2

# Hyperbolic specific parameters
CURVATURE=-1.0
MANIFOLD="lorentz"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.run $DISTRIBUTED_ARGS \
    pretrain_custom_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --manifold $MANIFOLD \
    --curvature $CURVATURE \
    --bf16