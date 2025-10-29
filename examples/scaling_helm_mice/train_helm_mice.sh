#!/bin/bash

# Training script for HELM-MiCE (renamed folder)

# Load modules and environment if needed
# module load cuda/11.8
# source activate megatron

# Directory configuration
CONFIG_PATH="examples/scaling_helm_mice/helm_mice_config.yaml"
CHECKPOINT_PATH="checkpoints/helm_mice"
DATA_PATH="/path/to/your/processed/data"
TENSORBOARD_PATH="tensorboard/helm_mice"
VOCAB_FILE="path/to/vocab.json"
MERGE_FILE="path/to/merges.txt"

# Model and training hyperparameters
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Training settings from your config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
SEQ_LEN=2048
HIDDEN_SIZE=910
NUM_LAYERS=16
NUM_HEADS=14
NUM_ROUTED_EXPERTS=8
NUM_SHARED_EXPERTS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.run $DISTRIBUTED_ARGS \
       pretrain_helm_mice.py \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LEN \
       --max-position-embeddings $SEQ_LEN \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --config-path $CONFIG_PATH \
       --bf16 \
       --scaled-upper-triang-masked-softmax-fusion \
       --bias-gelu-fusion \
       --rope \
       --mice-inter-dim 1820 \
       --num-routed-experts $NUM_ROUTED_EXPERTS \
       --num-shared-experts $NUM_SHARED_EXPERTS \
       --num-activated-experts 2 \
       --num-expert-groups 1 \
       --score-function softmax \
       --route-scale 1.0 \
       --train-curvature \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --distributed-backend nccl \
       --wandb-name "helm-mice-training" \
       --wandb-project "helm-mice" \
       --wandb-entity "your-entity"
