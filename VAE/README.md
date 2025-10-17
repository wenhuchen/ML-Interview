# Variational Autoencoder (VAE)

This directory contains a complete implementation of a Variational Autoencoder for image generation, demonstrating the fundamental concepts of probabilistic generative modeling and variational inference.

## Overview

Variational Autoencoders represent a cornerstone of probabilistic machine learning, combining neural networks with Bayesian inference to learn meaningful latent representations and generate new samples. This implementation focuses on MNIST digit generation while illustrating the key theoretical concepts.

## Files

### `vae.py` - Complete VAE Implementation
**Probabilistic Image Generation with MNIST**

A full-featured VAE implementation including training, inference, and generation capabilities:

## Core Architecture

### `VAE` Class Components

**Encoder Network:**
- **Convolutional Feature Extraction**: 3 conv layers with progressive channel increase (1→32→64→128)
- **Spatial Compression**: Stride-2 convolutions reduce 28×28 images to 7×7 feature maps  
- **Dense Projection**: Fully connected layer maps to 500-dimensional representation
- **Probabilistic Outputs**: Separate linear layers for mean (`μ`) and log-variance (`log σ²`)

**Decoder Network:**
- **Latent Projection**: Maps 20-dimensional latent code back to 500 dimensions
- **Spatial Reconstruction**: Unflatten to 7×7×128 feature maps
- **Deconvolutional Upsampling**: Transpose convolutions restore 28×28 resolution
- **Sigmoid Output**: Ensures pixel values in [0,1] range

**Latent Space:**
- **Dimensionality**: 20-dimensional continuous latent space
- **Prior Distribution**: Standard multivariate Gaussian N(0,I)
- **Reparameterization**: Enables backpropagation through stochastic sampling

## Mathematical Foundation

### Variational Lower Bound (ELBO)
The VAE optimizes the Evidence Lower BOund:
```
ELBO = 𝔼[log p(x|z)] - KL(q(z|x) || p(z))
```

**Components:**
1. **Reconstruction Term**: `𝔼[log p(x|z)]` - How well the decoder reconstructs input
2. **Regularization Term**: `KL(q(z|x) || p(z))` - How close the encoding is to prior

### Reparameterization Trick
Enables gradient flow through stochastic sampling:
```python
# Instead of sampling z ~ q(z|x) directly:
z = μ + σ ⊙ ε, where ε ~ N(0,I)
# This allows gradients to flow through μ and σ
```

### Loss Function Implementation
```python
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: KL(q(z|x) || p(z))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

## Key Features

### Training Mode
- **Full Training Loop**: 30 epochs on MNIST dataset
- **Batch Processing**: 128 samples per batch for efficient training
- **Progress Monitoring**: Real-time loss reporting and epoch tracking
- **Model Persistence**: Saves trained model with optimizer state

### Evaluation Mode
- **Latent Sampling**: Generates new digits by sampling from prior p(z)
- **Image Generation**: Decodes latent samples into realistic digit images
- **Visualization**: Creates and saves generated digit montages

### Probabilistic Generation Process
1. **Sample from Prior**: `z ~ N(0,I)` - sample random points in latent space
2. **Decode to Images**: Pass latent codes through decoder network
3. **Stochastic Output**: Generate diverse images from same latent region

## Variational Inference Concepts

### Encoder as Approximate Posterior
- **Recognition Network**: `q_φ(z|x)` approximates true posterior `p(z|x)`
- **Parametric Distribution**: Outputs parameters of diagonal Gaussian
- **Inference Network**: Learns to encode observations into latent distributions

### Decoder as Generative Model  
- **Generation Network**: `p_θ(x|z)` models likelihood of data given latent code
- **Reconstruction**: Maps latent codes back to observation space
- **Stochastic Generation**: Can produce multiple outputs from same latent input

### Latent Space Properties
- **Continuity**: Similar latent codes produce similar images
- **Interpolation**: Smooth transitions between different digit types
- **Disentanglement**: Different dimensions capture different aspects of variation

## Running the Code

### Training from Scratch:
```python
# In vae.py, set:
OPTION = 'train'
python vae.py
```
**Process:**
- Downloads MNIST dataset automatically
- Trains for 30 epochs with Adam optimizer
- Saves model to `vae_mnist_model.pth`
- Displays training progress and loss curves

### Generation Mode:
```python  
# In vae.py, set:
OPTION = 'eval'
python vae.py
```
**Process:**
- Loads pre-trained model from `vae_mnist_model.pth`
- Samples 10 random latent codes from N(0,I)
- Generates corresponding digit images
- Saves visualization to `image.png`

## Applications and Extensions

### Research Applications
- **Representation Learning**: Unsupervised discovery of meaningful features
- **Data Augmentation**: Generate additional training samples
- **Anomaly Detection**: Identify outliers through reconstruction error
- **Dimensionality Reduction**: Nonlinear alternative to PCA

### Advanced Variants
- **β-VAE**: Weighted KL term for better disentanglement
- **WAE**: Wasserstein distance instead of KL divergence  
- **VQ-VAE**: Discrete latent representations
- **Conditional VAE**: Class-conditional generation

## Key Insights

- **Probabilistic Framework**: VAEs provide principled approach to generative modeling
- **Latent Representations**: Learn meaningful, continuous representations of data
- **Trade-offs**: Balance between reconstruction quality and regularization
- **Scalability**: Architecture scales to high-dimensional data (images, text, audio)

This implementation demonstrates both the theoretical foundations and practical considerations essential for understanding modern generative modeling approaches.