#!/bin/bash

#SBATCH --job-name=GPU-torchrun
#SBATCH --mem=10G
#SBATCH --partition=rtx6000
#SBATCH --qos=m
#SBATCH --output=logs/%x.%j.log
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gpu[170-179]

export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
export MASTER_NODE=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`

# Set PyTorch distributed related variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Change to your network interface, e.g., "ens3", "bond0"
export NCCL_IB_DISABLE=1        # Disable InfiniBand if you don't use it
export NCCL_NET_GDR_LEVEL=0     # Disable GPU Direct RDMA

# Setting the master info
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29504

echo "--------------------------------- Running with Torch Run on $SLURM_NNODES nodes"
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=3 \
    run_with_torchrun.py