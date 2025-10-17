# Distributed Training with torchrun

This directory demonstrates various approaches to distributed training in PyTorch, with a focus on `torchrun` - PyTorch's modern distributed training launcher that simplifies multi-GPU and multi-node training setups.

## Overview

Distributed training enables scaling deep learning models across multiple GPUs and nodes, dramatically reducing training time for large models and datasets. This collection shows different methods for launching distributed training jobs, from simple multi-GPU setups to complex multi-node configurations.

## Files

### `train.py` - Training Script with Hugging Face Integration
**CNN Training on MNIST with Distributed Support**

A complete training pipeline that can be run in both single-GPU and distributed modes:

**Model Architecture:**
- **SimpleCNN**: Convolutional network for MNIST classification
- **Conv Layers**: 1→32→64 channels with ReLU activation and MaxPooling
- **FC Layers**: Flattened features → 128 → 10 output classes
- **Loss Integration**: Built-in loss computation for Trainer compatibility

**Key Features:**
- **Hugging Face Trainer**: Leverages the Trainer API for distributed training
- **Dataset Integration**: Uses Hugging Face datasets for MNIST loading
- **Custom Collation**: Handles image-label batching for CNN training
- **Automatic Evaluation**: Built-in evaluation metrics and logging

### `run_with_torchrun.py` - Distributed Process Setup
**Process Coordination and Environment Inspection**

Demonstrates how distributed processes are initialized and managed:

**Process Management:**
- **Environment Variables**: Reads `LOCAL_RANK`, `WORLD_SIZE` from torchrun
- **Process Group**: Initializes NCCL backend for GPU communication
- **Device Setting**: Maps each process to specific GPU device
- **Process Identification**: Shows hostname and rank information

**Key Concepts:**
- **LOCAL_RANK**: GPU ID on current node (0, 1, 2, 3...)
- **WORLD_SIZE**: Total number of processes across all nodes
- **NCCL Backend**: NVIDIA's optimized communication library for GPUs
- **Process Group**: Manages collective operations across distributed processes

### Additional Launch Scripts
The directory includes multiple launcher examples:
- **`run_with_torchrun.py`**: Modern torchrun approach
- **`run_with_mpirun.py`**: MPI-based distributed training
- **`run_with_python.py`**: Direct Python multiprocessing approach

## Distributed Training Concepts

### Data Parallel Training
**Most Common Distributed Strategy**

Each process maintains a complete copy of the model:
1. **Data Splitting**: Each process gets different batch of data
2. **Forward Pass**: Independent computation on each GPU
3. **Gradient Synchronization**: AllReduce to average gradients
4. **Parameter Update**: Synchronized updates across all processes

### torchrun Advantages
**Modern PyTorch Distributed Launcher**

**Key Benefits:**
- **Automatic Environment Setup**: Sets rank, world_size, master_addr automatically
- **Fault Tolerance**: Built-in restart mechanisms and error handling
- **Multi-node Support**: Seamless scaling from single-node to cluster
- **Elastic Training**: Dynamic scaling of worker processes

### Process Group Management
**Collective Communication Primitives**

PyTorch provides several communication patterns:
- **AllReduce**: Combine values across all processes (gradient averaging)
- **AllGather**: Collect tensors from all processes
- **Broadcast**: Send tensor from one process to all others
- **Reduce**: Combine values to single process

## Running Distributed Training

### Single Node, Multiple GPUs:
```bash
# Run on 4 GPUs with torchrun
torchrun --nproc_per_node=4 train.py

# Simple distributed test
torchrun --nproc_per_node=2 run_with_torchrun.py
```

**Expected Output:**
```
hostname$Process 0: 0 # 2
hostname$Process 0: Default device is set to 0
hostname$Process 1: 1 # 2  
hostname$Process 1: Default device is set to 1
```

### Multi-Node Training:
```bash
# Node 0 (master)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="10.0.0.1" --master_port=12345 train.py

# Node 1 (worker)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="10.0.0.1" --master_port=12345 train.py
```

### Alternative Launch Methods:
```bash
# MPI-based (if available)
mpirun -np 4 python run_with_mpirun.py

# Direct Python multiprocessing
python run_with_python.py
```

## Training Pipeline Features

### Hugging Face Integration
**Trainer API for Distributed Training**

The implementation leverages HF Trainer advantages:
- **Automatic DDP**: Wraps model in DistributedDataParallel automatically
- **Data Loading**: Handles distributed sampling automatically
- **Logging**: Synchronized logging across processes
- **Checkpointing**: Distributed-aware model saving/loading

### Custom Components
**CNN-Specific Adaptations**

**Custom Collate Function:**
```python
def collate_fn(batch):
    # Stack image tensors and label tensors separately
    pixel_values = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"image": pixel_values, "labels": labels}
```

**Model with Loss Integration:**
```python
def forward(self, image, labels=None):
    logits = self.network(image)
    if labels is not None:
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
    return {"logits": logits}
```

## Performance Considerations

### Communication Overhead
- **Gradient Synchronization**: AllReduce scaling with model size
- **Bandwidth Utilization**: Network topology affects multi-node performance
- **Communication/Computation Overlap**: Asynchronous gradient communication

### Memory Scaling
- **Linear Memory Scaling**: Each GPU maintains full model copy
- **Batch Size Scaling**: Effective batch size = local_batch × world_size
- **Gradient Accumulation**: Simulate larger batches without memory increase

### Efficiency Optimization
- **Mixed Precision**: Reduces communication and memory overhead
- **Gradient Compression**: Techniques like gradient quantization
- **Optimal Batch Sizing**: Balance between convergence and efficiency

## Common Patterns and Best Practices

### Environment Setup
```python
# Standard distributed training setup
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
```

### Error Handling
- **Timeout Configuration**: Handle slow network connections
- **Graceful Failures**: Proper cleanup on process failures
- **Logging Coordination**: Only log from rank 0 to avoid duplication

### Debugging Tips
- **Single Process Testing**: Test logic with WORLD_SIZE=1 first
- **Communication Verification**: Use simple tensor operations to verify setup
- **Process Coordination**: Ensure all processes reach collective operations

## Key Insights

- **torchrun Simplification**: Modern launcher eliminates manual environment setup complexity
- **Trainer Integration**: Hugging Face Trainer seamlessly handles distributed training details
- **Scalability**: Linear speedup achievable with proper network and workload configuration
- **Fault Tolerance**: Built-in mechanisms for handling distributed training failures

This implementation provides both the practical tools and theoretical understanding needed for effective distributed deep learning at scale.