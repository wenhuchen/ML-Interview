# Parallelism in Deep Learning

This directory contains implementations and examples of different parallelization strategies for deep learning models, focusing on distributed training techniques that enable scaling across multiple GPUs.

## Overview

Modern deep learning models often require multiple GPUs or nodes to train efficiently. This collection demonstrates three key parallelization approaches:

1. **Fully Sharded Data Parallel (FSDP)** - Memory-efficient data parallelism
2. **Model Parallelism** - Splitting model layers across devices 
3. **Tensor Parallelism** - Sharding individual tensors across devices

## Files

### `fsdp.py`
Demonstrates **Fully Sharded Data Parallel (FSDP)** using PyTorch's distributed training:

- Trains a CNN on MNIST dataset using FSDP
- Automatically shards model parameters and optimizer states across GPUs
- Reduces memory footprint compared to standard DataParallel
- Uses `size_based_auto_wrap_policy` for automatic model sharding
- Includes distributed data loading with `DistributedSampler`

**Key Concepts:**
- FSDP shards model parameters, gradients, and optimizer states
- Reduces GPU memory usage while maintaining performance
- Automatic gradient synchronization across ranks

### `model_parallelism_barebone_2_gpus.py` 
Shows basic **Model Parallelism** implementation:

- Splits a simple 2-layer MLP across 2 GPUs
- Layer 1 on GPU 0, Layer 2 on GPU 1  
- Manual forward/backward pass coordination using `dist.send/recv`
- Demonstrates explicit gradient communication between devices

**Key Concepts:**
- Pipeline parallelism: different layers on different devices
- Manual tensor communication between GPUs
- Synchronization of forward/backward passes

### `tensor_parallelism.py`
Advanced **Tensor Parallelism + FSDP** (2D Parallelism):

- Uses PyTorch's distributed tensor primitives
- Combines tensor parallelism (within nodes) with data parallelism (across nodes)
- Implements on a Llama2-style transformer model
- Demonstrates `ColwiseParallel`, `RowwiseParallel`, and `SequenceParallel` strategies

**Key Concepts:**
- 2D parallelism: TP within hosts, FSDP across hosts
- Attention and FFN layers sharded differently
- Device mesh topology for complex parallel strategies

### `library/llama2_model.py`
Supporting transformer implementation used by the tensor parallelism example.

## Parallelization Strategies Explained

### Data Parallelism
- Each GPU holds a full copy of the model
- Data batch is split across GPUs
- Gradients are averaged across replicas
- FSDP improves on this by sharding parameters

### Model Parallelism  
- Model layers are distributed across GPUs
- Sequential execution through the pipeline
- Useful when model is too large for single GPU

### Tensor Parallelism
- Individual weight matrices are sharded across GPUs
- Matrix operations executed in parallel
- Requires communication for certain operations
- Most efficient for large linear layers

## Running the Examples

Each script can be run using PyTorch's multiprocessing spawn:

```bash
# FSDP example
python fsdp.py --epochs 5

# Model parallelism (requires 2+ GPUs)  
python model_parallelism_barebone_2_gpus.py

# Tensor parallelism (requires 4+ GPUs)
torchrun --nproc_per_node=4 tensor_parallelism.py
```

## Requirements

- PyTorch with CUDA support
- Multiple GPUs (2+ for model parallelism, 4+ for tensor parallelism)
- NCCL backend for GPU communication

## Key Takeaways

- **FSDP**: Best for most use cases, reduces memory while keeping simplicity
- **Model Parallelism**: Good for very large models, but introduces pipeline bubbles
- **Tensor Parallelism**: Optimal for huge transformer layers, requires careful communication
- **2D Parallelism**: Combines approaches for maximum scalability