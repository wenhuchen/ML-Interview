# Mamba and State Space Models

This directory contains implementations of State Space Models (SSMs), focusing on the S4 (Structured State Space) architecture that forms the foundation of Mamba models.

## Overview

State Space Models represent a powerful alternative to transformers for sequence modeling, offering linear scaling with sequence length while maintaining strong modeling capabilities. This collection demonstrates the mathematical foundations and practical implementations of SSMs.

## Files

### `s4.py` - S4 (Structured State Space) Implementation
**Complete S4 Model Implementation**

This file provides a full implementation of the S4 architecture with both recurrent and convolutional modes:

**Core Components:**
- **HiPPO Matrix**: Uses HiPPO-LegS (Legendre Scaled) initialization for optimal memory retention
- **Discretization**: Zero-order hold (ZOH) method to convert continuous dynamics to discrete time
- **Dual Computation Modes**: Both recurrent and FFT-based convolution implementations

**Key Features:**
- `_hippo_matrix()`: Constructs the foundational HiPPO matrix for continuous-time dynamics
- `_discretize()`: Converts continuous system to discrete using matrix exponential
- `forward()`: Standard recurrent forward pass for shorter sequences
- `forward_conv()`: FFT-based convolution for efficient long sequence processing

**Mathematical Foundation:**
```python
# Continuous-time system: dx/dt = Ax + Bu, y = Cx
# Discrete-time system: x[k+1] = A_d x[k] + B_d u[k], y[k] = C x[k]
# Where A_d = exp(A*Δt), B_d = A^(-1)(A_d - I)B
```

### `state_space_model.py` - Mathematical Foundations
**Educational Examples and Verification**

Demonstrates core concepts underlying state space models:

**Functions:**
1. **`verify_convolution_theorem()`**
   - Validates the mathematical equivalence between time-domain convolution and frequency-domain multiplication
   - Shows how FFT-based methods provide computational advantages
   - Visualizes both approaches with identical results

2. **`state_space_model()`**
   - Simple 2-state system simulation showing basic SSM dynamics
   - Demonstrates state evolution over time with constant input
   - Plots state trajectories and system output

**Educational Value:**
- **Convolution Theorem**: Foundation for efficient SSM computation via FFT
- **State Evolution**: How internal states track and summarize input history
- **Linear Systems**: Basic principles that extend to more complex architectures

## State Space Model Concepts

### Mathematical Formulation
State space models are defined by the system:
```
dx/dt = Ax(t) + Bu(t)    (continuous time)
y(t) = Cx(t) + Du(t)

x[k+1] = A_d x[k] + B_d u[k]    (discrete time)
y[k] = C x[k] + D u[k]
```

### HiPPO (High-order Polynomial Projection Operator)
- **Purpose**: Optimal way to compress infinite history into finite state
- **HiPPO-LegS**: Uses Legendre polynomials for memory-efficient sequence processing
- **Matrix Structure**: Lower triangular with specific entries for polynomial basis

### Key Advantages over Transformers
1. **Linear Scaling**: O(L) complexity vs O(L²) for attention
2. **Constant Memory**: Fixed state size regardless of sequence length
3. **Hardware Efficiency**: More parallelizable and cache-friendly
4. **Long Sequences**: Can process very long sequences without quadratic blowup

### S4 Innovations
- **Structured Matrices**: Exploits mathematical structure for efficiency
- **Stable Discretization**: Careful numerical methods prevent instability
- **Dual Modes**: Recurrent for inference, convolution for training

## Mamba Architecture Connection

While this directory focuses on S4 fundamentals, these concepts directly lead to Mamba:

1. **Selective SSMs**: Mamba makes state transitions input-dependent
2. **Hardware Awareness**: Optimized for modern GPU memory hierarchies  
3. **Scaling**: Enables transformer-competitive performance at linear cost

## Running the Code

### S4 Model:
```bash
python s4.py
```
**Output:**
- Demonstrates both recurrent and convolutional forward passes
- Shows output sequences from random inputs
- Verifies numerical consistency between methods

### State Space Fundamentals:
```bash
python state_space_model.py
```
**Output:**
- `convolution_theorem.png`: Visual verification of convolution theorem
- `state_space_model.png`: State evolution and output plots
- Numerical verification of time vs frequency domain equivalence

## Requirements

- NumPy for numerical computation
- SciPy for linear algebra operations (`linalg.expm`)
- Matplotlib for visualization

## Key Insights

- **Mathematical Rigor**: SSMs provide principled approach to sequence modeling
- **Computational Efficiency**: FFT enables linear-time training despite recurrent structure
- **Memory Management**: Fixed-size state provides constant memory usage
- **Foundation for Innovation**: S4 principles enable architectures like Mamba

This implementation provides both theoretical understanding and practical experience with the mathematical foundations underlying modern efficient sequence models.