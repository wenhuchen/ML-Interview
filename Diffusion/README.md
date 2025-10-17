# Diffusion Models for Image Generation

This directory implements a complete diffusion model for image generation, demonstrating the DDPM (Denoising Diffusion Probabilistic Models) approach that has revolutionized generative AI and enabled models like DALL-E 2, Midjourney, and Stable Diffusion.

## Overview

Diffusion models generate images by learning to reverse a noise corruption process. They start with pure noise and gradually denoise it over many steps to produce high-quality samples. This approach has proven highly effective for generating diverse, high-fidelity images.

## Files

### `diffusion.py` - Complete DDPM Training and Sampling
**End-to-End Diffusion Model Implementation**

The main file containing the full diffusion model pipeline including training, sampling, and evaluation.

### `mnist_unet.py` - U-Net Architecture for Denoising
**Specialized Neural Network for Diffusion**

A custom U-Net implementation optimized for the denoising task in diffusion models.

## Core Components

### `LinearBetaScheduler` Class
**Noise Schedule Management**

Controls the noise addition/removal process across timesteps:

**Key Components:**
- **Beta Schedule**: Linear interpolation from `β_start=0.0001` to `β_end=0.02`
- **Alpha Schedule**: Computes `α_t = 1 - β_t` and cumulative products `ᾱ_t`
- **Variance Computation**: Manages noise levels at each timestep

**Mathematical Foundation:**
```python
# Forward process (adding noise):
# q(x_t|x_{t-1}) = N(x_t; √α_t x_{t-1}, β_t I)
# q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)

# Reverse process (denoising):  
# p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
```

### U-Net Denoising Network

**Architecture Overview:**
- **Encoder-Decoder Structure**: Downsampling followed by upsampling
- **Skip Connections**: Preserve fine-grained details across scales
- **Time Embedding**: Injects timestep information into each layer
- **Residual Blocks**: Based on ShuffleNet v2 for efficiency

**Key Features:**
- **Multi-scale Processing**: Handles different resolution features
- **Time-Conditional**: Network output depends on current timestep
- **Efficient Design**: Optimized for fast training and inference

### Training Process

**Loss Function:**
The model is trained to predict the noise added at each step:
```python
# Simplified training objective:
# L = E[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]
# where ε is the noise and ε_θ is the predicted noise
```

**Training Pipeline:**
1. **Random Timestep**: Sample t uniformly from [0, T]
2. **Noise Addition**: Add Gaussian noise to clean images
3. **Noise Prediction**: Train network to predict the added noise
4. **Loss Computation**: MSE between actual and predicted noise

### Sampling Algorithms

The implementation includes two sampling strategies:

#### `sample()` - Standard DDPM Sampling
**Original DDPM Reverse Process**
- Gradually removes noise over T timesteps
- Uses learned noise prediction to estimate clean image
- Includes stochastic sampling for diversity

#### `sample_clip()` - Improved Sampling with Clipping  
**Enhanced Sampling Strategy**
- Predicts clean image directly: `x_0_pred = (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t`
- Applies clipping: `x_0_pred.clamp_(-1, 1)` for stability
- Uses predicted clean image for more stable reverse process

## Advanced Features

### Time Embedding Architecture
**Temporal Information Integration**

The U-Net incorporates timestep information through:
- **Sinusoidal Position Embedding**: Maps timesteps to high-dimensional vectors
- **Time MLP**: Processes time embeddings and injects into feature maps
- **Broadcast Addition**: Adds time information across spatial dimensions

### Multi-Scale Processing
**Hierarchical Feature Learning**

The U-Net processes images at multiple scales:
- **Encoder Path**: Progressive downsampling (28×28 → 14×14 → 7×7)
- **Bottleneck**: Processes lowest resolution features
- **Decoder Path**: Progressive upsampling with skip connections
- **Feature Fusion**: Combines multi-scale information for final output

### Training Infrastructure

**Experiment Tracking:**
- **Weights & Biases Integration**: Comprehensive experiment logging
- **Learning Rate Scheduling**: OneCycleLR for optimal training dynamics
- **Model Checkpointing**: Saves trained models for inference

**Data Pipeline:**
- **MNIST Dataset**: Normalized to [-1, 1] range for stability
- **Data Augmentation**: Built-in through the stochastic training process
- **Efficient Loading**: Optimized data loading with multiple workers

## Running the Code

### Training Mode:
```bash
python diffusion.py --mode train
```

**Training Process:**
- Downloads MNIST dataset automatically  
- Trains U-Net to predict noise for 30 epochs
- Logs training progress to Weights & Biases
- Saves trained model checkpoints

### Generation Mode:
```bash
python diffusion.py --mode eval
```

**Generation Process:**
- Loads pre-trained diffusion model
- Starts with random noise tensor
- Applies T-step denoising process
- Saves generated images to disk

### Sampling Visualization:
```python
# Generate multiple samples
save_image(size=16, model)  # Creates 4x4 grid of generated digits
```

## Diffusion Model Theory

### Forward Process (Noise Addition)
The forward process gradually adds Gaussian noise:
```
x_1, x_2, ..., x_T ~ q(x_t|x_{t-1}) = N(√α_t x_{t-1}, β_t I)
```

### Reverse Process (Denoising)  
The neural network learns to reverse this process:
```
x_{T-1}, ..., x_1, x_0 ~ p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), σ_t² I)
```

### Key Insights
- **Tractable Training**: Forward process allows exact noise level computation
- **Gradual Generation**: Reverse process creates images through gradual denoising
- **High Quality**: Multiple denoising steps enable fine detail generation
- **Diversity**: Stochastic sampling produces varied outputs

## Applications and Extensions

### Research Directions
- **Conditional Generation**: Text-to-image, class-conditional models
- **Faster Sampling**: DDIM, DPM-Solver for fewer steps
- **Higher Resolution**: Progressive training, attention mechanisms
- **Different Domains**: Audio, video, 3D generation

### Practical Applications
- **Content Creation**: Art, design, creative applications
- **Data Augmentation**: Synthetic training data generation
- **Image Editing**: Inpainting, super-resolution, style transfer
- **Scientific Modeling**: Molecular design, weather simulation

## Key Insights

- **Iterative Refinement**: Quality improves through gradual denoising process
- **Scale of Training**: Large datasets and computation enable impressive results  
- **Architecture Matters**: U-Net design crucial for effective denoising
- **Sampling Trade-offs**: Quality vs speed considerations in deployment

This implementation provides a solid foundation for understanding diffusion models and can be extended to more complex domains and applications.