# Gradient Checkpointing

This directory implements gradient checkpointing (also known as activation checkpointing), a memory optimization technique that enables training of much larger neural networks by trading compute for memory during backpropagation.

## Overview

Gradient checkpointing reduces memory usage during training by not storing all intermediate activations during the forward pass. Instead, it recomputes them as needed during backpropagation, allowing for significant memory savings at the cost of additional forward passes.

## Files

### `run.py` - Complete Gradient Checkpointing Implementation
**Memory-Efficient Training with Activation Recomputation**

A comprehensive implementation that includes both the checkpointing mechanism and a practical demonstration of memory savings.

## Core Implementation

### `CheckpointFunction` Class
**Custom PyTorch Autograd Function**

Implements the core checkpointing logic using PyTorch's autograd system:

**Forward Pass (`forward`):**
- **RNG State Preservation**: Optionally saves random number generator state for deterministic recomputation
- **Function Storage**: Saves the computation function and inputs for backward pass
- **No-Grad Execution**: Runs forward pass without storing intermediate activations
- **Memory Efficiency**: Only keeps final outputs, discarding intermediate results

**Backward Pass (`backward`):**
- **RNG State Restoration**: Restores saved random state for identical recomputation
- **Input Detachment**: Detaches inputs from computation graph to avoid double-counting
- **Recomputation**: Re-executes forward pass with gradients enabled
- **Gradient Computation**: Computes gradients using recomputed activations

### `checkpoint()` Function
**User-Friendly Interface**

Provides a simple API for applying checkpointing to any function:
```python
def checkpoint(run_fn, *args, preserve_rng_state=True):
    return CheckpointFunction.apply(run_fn, preserve_rng_state, *args)
```

**Key Parameters:**
- **`run_fn`**: Function to be checkpointed
- **`*args`**: Input tensors to the function
- **`preserve_rng_state`**: Whether to preserve randomness across recomputation

### `CheckpointedMLP` Class
**Practical Demonstration Model**

A multi-layer perceptron that demonstrates gradient checkpointing in practice:

**Architecture:**
- **Input Layer**: Linear transformation with ReLU and Dropout
- **Hidden Layers**: 4 identical layers with same hidden dimension
- **Configurable Checkpointing**: Can toggle checkpointing on/off for comparison

**Checkpointing Strategy:**
- **Modular Functions**: Each layer implemented as separate checkpointable function
- **Selective Application**: Only middle layers are checkpointed
- **Memory vs Compute Trade-off**: Demonstrates practical impact of checkpointing

## Gradient Checkpointing Theory

### Memory-Compute Trade-off
**Fundamental Principle**

Traditional backpropagation stores all intermediate activations:
```
Memory Usage: O(L × B × H)  where L=layers, B=batch_size, H=hidden_size
Compute: 1 forward + 1 backward pass
```

Gradient checkpointing reduces memory at cost of recomputation:
```
Memory Usage: O(√L × B × H)  (optimal checkpointing strategy)
Compute: 1 forward + ~1.5 forward passes (for recomputation)
```

### Recomputation Process
**How Backward Pass Works**

1. **Forward Pass**: Only final layer outputs are kept
2. **Backward Trigger**: Gradient computation begins from loss
3. **Recomputation**: Forward pass re-executed to get intermediate activations
4. **Gradient Calculation**: Standard backprop using recomputed activations
5. **Memory Release**: Recomputed activations discarded after use

### RNG State Management
**Maintaining Determinism**

Critical for layers with randomness (dropout, stochastic regularization):
- **State Capture**: Save RNG state before forward pass
- **State Restoration**: Restore exact same RNG state during recomputation
- **Identical Results**: Ensures recomputed activations match original values

## Implementation Details

### `detach_variable()` Function
**Proper Gradient Flow Management**

Handles tensor detachment while preserving gradient requirements:
```python
def detach_variable(inputs):
    # Detach from computation graph but preserve requires_grad flag
    x = inp.detach()
    x.requires_grad = inp.requires_grad
    return x
```

**Purpose:**
- **Graph Separation**: Prevents double-counting in gradient computation
- **Gradient Preservation**: Maintains gradient flow for subsequent operations
- **Memory Safety**: Ensures proper cleanup of computation graph

### Memory Measurement
**Practical Performance Analysis**

The implementation includes GPU memory tracking:
```python
torch.cuda.reset_peak_memory_stats()
# ... training code ...
peak_memory = torch.cuda.max_memory_allocated()
```

**Comparison Framework:**
- **With Checkpointing**: `python run.py True`
- **Without Checkpointing**: `python run.py False`
- **Memory Reporting**: Shows peak GPU memory usage for each mode

## Running the Code

### Memory Comparison Test:
```bash
# Without gradient checkpointing
python run.py False

# With gradient checkpointing  
python run.py True
```

**Expected Output:**
```
Peak GPU memory usage: XXXX.XX MB with Checkpoint = False
Peak GPU memory usage: YYYY.YY MB with Checkpoint = True
```

**Typical Results:**
- **Memory Reduction**: 30-70% less peak memory usage with checkpointing
- **Training Time**: 20-50% longer due to recomputation overhead
- **Model Size Impact**: Greater savings for larger models with more layers

### Understanding the Results
**Memory Usage Analysis**

The implementation demonstrates:
- **Baseline Memory**: Full activation storage without checkpointing
- **Checkpointed Memory**: Reduced memory with selective activation storage
- **Trade-off Quantification**: Exact memory vs compute trade-off measurement

## Advanced Applications

### Optimal Checkpointing Strategies
**Beyond Basic Implementation**

Advanced strategies for large models:
- **Square Root Rule**: Checkpoint every √L layers for optimal trade-off
- **Gradient Accumulation**: Combine with gradient accumulation for further savings
- **Mixed Strategies**: Checkpoint some layers, store others based on memory budget

### Integration with Large Models
**Practical Deployment Scenarios**

Common use cases:
- **Transformer Models**: Checkpoint attention blocks in large language models
- **Computer Vision**: Checkpoint ResNet blocks in deep CNN architectures
- **Memory-Constrained Training**: Enable larger batch sizes or model sizes

### Production Considerations
**Real-World Usage Patterns**

- **PyTorch Integration**: `torch.utils.checkpoint` provides production-ready implementation
- **Framework Support**: Hugging Face Transformers includes automatic checkpointing
- **Debugging**: Checkpoint-aware debugging tools and profilers

## Key Insights

- **Memory Scaling**: Enables training models that wouldn't fit in GPU memory otherwise
- **Compute Overhead**: ~50% additional compute cost is often worthwhile for memory savings
- **Implementation Complexity**: Requires careful handling of random states and gradient flow
- **Practical Impact**: Critical technique for scaling deep learning to larger models and batch sizes

## Best Practices

- **Selective Application**: Not all layers need checkpointing - profile to find optimal strategy
- **RNG State Management**: Always preserve random states for layers with stochastic operations
- **Memory Profiling**: Use GPU memory profilers to validate memory savings
- **Testing**: Verify numerical equivalence between checkpointed and non-checkpointed training

This implementation provides both the theoretical foundation and practical tools needed to effectively apply gradient checkpointing in memory-constrained deep learning scenarios.