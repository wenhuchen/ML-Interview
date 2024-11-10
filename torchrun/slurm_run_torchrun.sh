#!/bin/bash

#SBATCH --job-name=GPU-torchrun
#SBATCH --mem=10G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --output=logs/%x.%j.log
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=gpu[170-179]

export NODELIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
export MASTER_NODE=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n1`

# Set PyTorch distributed related variables
export NCCL_DEBUG=INFO
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29504

echo "--------------------------------- Running with Torch Run on $SLURM_NNODES nodes"
srun torchrun \
    --nproc_per_node=$SLURM_NNODES \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_with_torchrun.py