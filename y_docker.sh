docker run --runtime=nvidia --gpus all -it  --name=aosong_hypermega \
  -v /fsx/ubuntu/users/aosong/ft/hyper/Megatron-LM:/workspace/megatron \
  -v /fsx/ubuntu/users/aosong/data:/workspace/dataset \
  -v /opt/dlami/nvme/aosong/checkpoints/hyper:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
