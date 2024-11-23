#!/bin/bash

#SBATCH --job-name=GPU-mpi
#SBATCH --mem=10G
#SBATCH --partition=rtx6000
#SBATCH --qos=m
#SBATCH --output=logs/%x.%j.log
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8

echo "--------------------------------- Running with MPI Run"
mpirun python run_with_mpirun.py
