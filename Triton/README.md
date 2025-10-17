# Triton GPU Kernels

This directory demonstrates the power of Triton, a domain-specific language for writing efficient GPU kernels that can match or exceed the performance of hand-optimized CUDA while being significantly more accessible to Python developers.

## Overview

Triton enables developers to write GPU kernels in Python-like syntax that compile to highly optimized machine code. This approach bridges the gap between easy-to-use frameworks like PyTorch and low-level CUDA programming, making high-performance GPU computing more accessible.

## Files

### `layer_norm.py` - Fused Layer Normalization Kernel
**High-Performance Layer Normalization Implementation**

This file demonstrates a fused bias addition + layer normalization kernel that showcases several advanced Triton concepts:

## Core Implementation

### `fused_bias_ln` Kernel
**Mathematical Operations:**
1. **Bias Addition**: `x = X + BIAS`
2. **Mean Computation**: `mean = Σx / N`  
3. **Variance Calculation**: `var = Σx² / N - mean²`
4. **Normalization**: `y = (x - mean) * rsqrt(var + eps) * γ + β`

**Key Features:**
- **Single Kernel Launch**: All operations fused into one GPU kernel call
- **Memory Efficiency**: Minimizes memory bandwidth by avoiding intermediate storage
- **Vectorized Operations**: Leverages Triton's automatic vectorization
- **Template Specialization**: Uses `tl.constexpr` for compile-time optimization

### Triton Programming Concepts

**Program Structure:**
```python
@triton.jit
def kernel_name(inputs, outputs, constants: tl.constexpr):
    # Get program/thread ID
    pid = tl.program_id(axis=0)
    
    # Calculate memory offsets
    offs = pid * N + tl.arange(0, N)
    
    # Load data vectorized
    x = tl.load(X + offs)
    
    # Perform computations
    result = compute(x)
    
    # Store results
    tl.store(Y + offs, result)
```

**Key Language Features:**
- **`@triton.jit`**: Just-in-time compilation decorator
- **`tl.constexpr`**: Compile-time constants for optimization
- **`tl.program_id()`**: Gets the current program instance ID
- **`tl.arange()`**: Generates ranges of indices for vectorization
- **`tl.load/store()`**: Memory access primitives with bounds checking

### Performance Optimizations

**Memory Access Patterns:**
- **Coalesced Access**: All threads in a warp access contiguous memory
- **Vectorization**: `tl.arange(0, N)` enables SIMD operations
- **Reduced Trips**: Single kernel launch instead of multiple PyTorch operations

**Computational Efficiency:**
- **Mathematical Optimizations**: `rsqrt()` instead of `1/sqrt()` for better performance
- **Fused Operations**: Eliminates intermediate memory allocations
- **Compile-time Constants**: `tl.constexpr` enables aggressive optimization

### Benchmarking and Verification

**Correctness Testing:**
- Compares Triton output against PyTorch's `F.layer_norm()`
- Validates numerical accuracy with absolute difference measurement

**Performance Measurement:**
- Warmup compilation phase (first call)
- Timed execution over 100 iterations
- CUDA synchronization for accurate timing

## Triton vs CUDA vs PyTorch

### Traditional PyTorch Implementation
```python
# Multiple kernel launches, intermediate tensors
x_bias = x + bias
x_normalized = F.layer_norm(x_bias, (N,), gamma, beta, eps)
```

### Equivalent CUDA (Conceptual)
```cpp
// Hundreds of lines of CUDA C++
// Manual memory management
// Explicit thread/block management
// Complex optimization considerations
```

### Triton Implementation
```python
# Single fused kernel
# Python-like syntax
# Automatic optimization
# ~20 lines of readable code
```

## Key Advantages of Triton

### Developer Productivity
- **Python Syntax**: Familiar programming model for ML researchers
- **Automatic Optimization**: Compiler handles low-level optimizations
- **Rapid Prototyping**: Fast iteration on kernel designs
- **Integration**: Seamless interop with PyTorch

### Performance Benefits  
- **Fusion Opportunities**: Easy to combine multiple operations
- **Memory Bandwidth**: Optimal memory access patterns
- **Computation Efficiency**: Modern GPU instruction utilization
- **Scalability**: Performance across different GPU architectures

### Accessibility
- **Lower Barrier**: No need for CUDA expertise
- **Maintainability**: Readable, concise implementations  
- **Debugging**: Better error messages and debugging support
- **Community**: Growing ecosystem of optimized kernels

## Running the Code

### Basic Execution:
```bash
python layer_norm.py
```

**Expected Output:**
```
max diff 1.1920928955078125e-07
Triton fused: X.XXX ms
```

**Interpretation:**
- **Max Diff**: Numerical difference vs PyTorch (should be ~1e-7)
- **Execution Time**: Performance in milliseconds for fused operation

### Customization Options:
```python
# Modify parameters in run_once()
run_once(B=512, N=8192, dtype=torch.float32, eps=1e-6)
```

## Advanced Applications

Triton enables optimization of many ML kernels:
- **Attention Mechanisms**: Fused multi-head attention
- **Activation Functions**: Custom activation implementations
- **Matrix Operations**: Specialized GEMM variants
- **Normalization**: Group norm, RMS norm, etc.
- **Element-wise**: Complex broadcasting operations

## Key Insights

- **Abstraction Level**: Perfect balance between ease-of-use and performance control
- **Compiler Technology**: Advanced optimization without manual tuning
- **Memory Hierarchy**: Automatically leverages GPU memory subsystems
- **Future-Proof**: Portable across different GPU generations and vendors

This implementation demonstrates how Triton democratizes high-performance GPU programming, enabling ML practitioners to achieve CUDA-level performance with Python-level simplicity.