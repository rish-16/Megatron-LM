#!/bin/bash

# Distributed training script for custom GPT model

# Number of GPUs to use
GPUS_PER_NODE=8
# Number of nodes
NNODES=1

# Directory configuration
CONFIG_PATH="examples/custom_gpt/config.yaml"
CHECKPOINT_PATH="checkpoints/custom_gpt"
VOCAB_FILE="path/to/your/vocab.json"  # Update this
MERGE_FILE="path/to/your/merges.txt"  # Update this
DATA_PATH="path/to/your/data"         # Update this

# Model and training hyperparameters
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128
TP_SIZE=8  # Tensor parallel size
PP_SIZE=1  # Pipeline parallel size

python pretrain_custom_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
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
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --config-path $CONFIG_PATH