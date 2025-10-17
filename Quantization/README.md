# Model Quantization with AWQ

This directory implements AWQ (Activation-aware Weight Quantization), a state-of-the-art technique for compressing neural networks while maintaining performance. The implementation demonstrates how to reduce model size significantly with minimal accuracy degradation.

## Overview

Model quantization reduces the precision of neural network weights and activations, enabling deployment on resource-constrained devices while maintaining competitive performance. AWQ specifically focuses on preserving important weights based on activation patterns, offering superior accuracy-compression trade-offs.

## Files

### `awq.py` - AWQ Implementation and Testing
**Complete Weight Quantization Pipeline**

Demonstrates end-to-end quantization of a neural network using AWQ methodology:

## Core Components

### `SimpleNN` Class
**Original Full-Precision Network**

A straightforward feedforward network for testing quantization:
- **Architecture**: 64 → 128 → 256 → 128 → 10 (configurable dimensions)
- **Activation**: ReLU between hidden layers
- **Output**: Raw logits for classification
- **No Bias**: Simplified design focusing on weight quantization

### `QuantizedNN` Class  
**Post-Quantization Network Wrapper**

Manages quantized model components:
- **Quantized Layers**: Converted low-precision weight matrices
- **Scale Information**: Metadata for activation scaling
- **Identical Forward**: Same computation graph as original network

### AWQ Algorithm Implementation

**Key Functions:**

1. **`compute_activation_stats()`** (from helper.py):
   - Collects activation statistics during forward passes
   - Measures channel-wise activation magnitudes
   - Identifies important activation channels for weight preservation

2. **`search_module_scale()`** (from helper.py):
   - Finds optimal scaling factors for each layer
   - Balances quantization error with activation importance
   - Returns quantized weights and scaling metadata

3. **`quantize_model()`**:
   - Orchestrates the complete quantization pipeline
   - Applies AWQ to each linear layer sequentially
   - Preserves activation functions and network structure

## AWQ Methodology

### Activation-Aware Quantization
Unlike naive quantization approaches that treat all weights equally, AWQ:

**Core Insight**: Weights corresponding to large activation channels contribute more to model output and should be preserved with higher precision.

**Mathematical Foundation**:
```
For layer output Y = XW:
- Measure activation magnitudes ||X[:, i]|| for each channel i
- Scale quantization based on activation importance
- Preserve high-impact weights, aggressively quantize low-impact ones
```

### Quantization Process
1. **Calibration**: Run representative data through the network
2. **Statistics Collection**: Measure per-channel activation magnitudes  
3. **Scale Search**: Find optimal scaling factors for weight quantization
4. **Weight Conversion**: Convert weights to low-precision representation
5. **Metadata Storage**: Save scaling information for inference

## Key Features

### Comprehensive Testing Framework
- **Synthetic Data**: Generates realistic test data for validation
- **Weight Comparison**: Detailed analysis of quantization effects per layer
- **Output Verification**: Compares model outputs before/after quantization
- **Size Analysis**: Calculates compression ratios achieved

### Quantization Analysis
The implementation provides detailed metrics:

**Weight Error Analysis**:
- Layer-by-layer weight difference computation
- Accumulation of quantization errors across the network
- Identification of layers most affected by quantization

**Model Size Reduction**:
- Calculates exact memory savings from quantization
- Reports compression ratios (typically 50-75% size reduction for 8-bit)
- Accounts for different bit-widths (4-bit, 8-bit options)

### Practical Considerations
- **GPU Compatibility**: Designed for CUDA execution
- **Configurable Precision**: Supports different quantization bit-widths
- **Preservation of Architecture**: Maintains original network structure
- **Inference Efficiency**: Quantized models suitable for deployment

## Running the Code

### Basic Quantization Test:
```bash
python awq.py
```

**Expected Output Flow**:
1. **Original Weights**: Display full-precision weight matrices
2. **Quantization Process**: AWQ algorithm execution
3. **Quantized Weights**: Show post-quantization weight values  
4. **Error Analysis**: Per-layer quantization error accumulation
5. **Output Comparison**: Before/after quantization inference results
6. **Compression Stats**: Model size reduction percentage

### Customization Options:
```python
# Modify quantization parameters
quantization_bits = 4  # More aggressive compression
input_dim = 128       # Larger network
hidden_dims = [256, 512, 256, 10]  # Different architecture
```

## AWQ vs Other Quantization Methods

### Post-Training Quantization (PTQ)
- **Uniform Scaling**: Applies same quantization to all weights
- **Limited Accuracy**: Can suffer significant performance degradation  
- **Simple Implementation**: Straightforward to apply

### Quantization-Aware Training (QAT)
- **Training Integration**: Simulates quantization during training
- **High Accuracy**: Can achieve excellent results
- **Computational Cost**: Requires full retraining

### AWQ (Activation-Aware Weight Quantization)
- **Selective Quantization**: Preserves important weights based on activations
- **Post-Training**: No retraining required
- **Optimal Trade-off**: Better accuracy than PTQ, faster than QAT

## Applications and Benefits

### Deployment Advantages
- **Memory Reduction**: 2-8x smaller model sizes
- **Inference Speed**: Faster computation with low-precision arithmetic
- **Energy Efficiency**: Reduced power consumption on mobile devices
- **Hardware Support**: Better utilization of specialized inference chips

### Use Cases
- **Mobile Deployment**: Fitting large models on smartphones/tablets
- **Edge Computing**: IoT devices with limited resources  
- **Server Optimization**: Higher throughput in data centers
- **Real-time Applications**: Reduced latency for time-critical systems

## Key Insights

- **Activation Awareness**: Leveraging activation patterns leads to better quantization decisions
- **Selective Preservation**: Not all weights are equally important for model performance
- **Calibration Data**: Quality of calibration dataset significantly impacts results
- **Layer Sensitivity**: Different layers show varying sensitivity to quantization

This implementation demonstrates how modern quantization techniques can dramatically reduce model size while maintaining competitive accuracy, making deployment of large neural networks feasible across diverse computing environments.