# Reinforcement Learning - GAE Implementation

This directory implements Generalized Advantage Estimation (GAE), a crucial technique in modern reinforcement learning that provides a practical solution for estimating advantages in policy gradient methods.

## Overview

Generalized Advantage Estimation addresses the fundamental bias-variance trade-off in policy gradient methods by providing a parameterized family of advantage estimators. This technique is essential for stable and efficient training of modern RL algorithms like PPO, TRPO, and A3C.

## Files

### `run.py` - GAE Computation Implementation
**Generalized Advantage Estimation Algorithm**

A clean implementation of the GAE algorithm with detailed computation steps:

## Core Implementation

### `compute_advantages()` Function
**GAE Algorithm Implementation**

```python
def compute_advantages(rewards, values, gamma, lambda_):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = lastgaelam = delta + gamma * lambda_ * lastgaelam
    return advantages
```

**Key Components:**
- **Temporal Difference Error**: `δ_t = r_t + γV(s_{t+1}) - V(s_t)`
- **GAE Recursion**: `A_t^{GAE} = δ_t + γλA_{t+1}^{GAE}`
- **Bootstrap Value**: Uses value function estimate for next state
- **Terminal Handling**: Properly handles episode termination

## GAE Theory and Motivation

### The Bias-Variance Trade-off
**Fundamental Challenge in Policy Gradients**

Policy gradient methods need advantage estimates `A(s,a)` to determine which actions were better than expected:

**High Bias, Low Variance** (1-step TD):
```
A_t = r_t + γV(s_{t+1}) - V(s_t)
```

**Low Bias, High Variance** (Monte Carlo):
```
A_t = Σ_{i=0}^{∞} γ^i r_{t+i} - V(s_t)
```

**GAE Compromise**:
```
A_t^{GAE(γ,λ)} = Σ_{i=0}^{∞} (γλ)^i δ_{t+i}
```

### Mathematical Foundation

#### Temporal Difference Error
The foundation of GAE is the TD error:
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

This represents the "surprise" - how much better or worse the actual transition was compared to our value function's prediction.

#### GAE Formula
GAE combines multiple TD errors with exponentially decaying weights:
```
A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

**Parameters:**
- **γ (gamma)**: Discount factor for future rewards
- **λ (lambda)**: GAE parameter controlling bias-variance trade-off

#### Recursive Implementation
The clever recursive formulation enables efficient computation:
```
A_t^{GAE} = δ_t + γλA_{t+1}^{GAE}
```

This allows backward iteration through the episode, building advantages incrementally.

## Parameter Analysis

### Lambda (λ) Parameter Effects

**λ = 0** (High Bias, Low Variance):
- Reduces to 1-step TD error: `A_t = δ_t`
- Fast learning but may miss long-term dependencies
- More stable but potentially suboptimal policies

**λ = 1** (Low Bias, High Variance):  
- Approaches Monte Carlo estimation
- Captures full reward sequence but noisy
- Slower learning due to high variance

**λ ∈ (0,1)** (Balanced Trade-off):
- Most common choice: λ = 0.95
- Balances bias and variance effectively
- Enables stable learning with good performance

### Gamma (γ) Parameter Effects

**γ → 0**: Only immediate rewards matter
**γ → 1**: All future rewards weighted equally
**γ = 0.99**: Common choice for most RL problems

## Implementation Details

### Backward Iteration
**Why Process Episodes in Reverse**

```python
for t in reversed(range(len(rewards))):
    # Process from last timestep to first
    # This allows us to use already computed A_{t+1}
```

**Benefits:**
- **Efficiency**: Each advantage computed once using previous result
- **Correctness**: Ensures proper dependency chain
- **Memory**: Constant memory usage regardless of episode length

### Terminal State Handling
**Proper Episode Boundary Management**

```python
if t == len(rewards) - 1:
    next_value = 0  # Terminal state has no future value
else:
    next_value = values[t + 1]  # Use value function estimate
```

**Critical Details:**
- **Terminal States**: Set next_value = 0 for episode end
- **Value Bootstrapping**: Use value estimates for non-terminal transitions
- **Episode Boundaries**: Proper handling prevents value bleeding between episodes

## Example Walkthrough

### Sample Data Analysis
**Understanding the Computation**

Given:
```python
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]  # Sparse reward at end
values  = [0.9, 0.8, 0.7, 0.3, 0.5]  # Value function estimates
gamma = 0.99, lambda = 0.95
```

**Step-by-Step Computation:**

**t=4 (Terminal):**
- `δ_4 = 1.0 + 0.99 × 0 - 0.5 = 0.5`
- `A_4 = 0.5` (base case)

**t=3:**
- `δ_3 = 0.0 + 0.99 × 0.5 - 0.3 = 0.195`
- `A_3 = 0.195 + 0.99 × 0.95 × 0.5 = 0.665`

**Continue backward...**

### Result Interpretation
**What GAE Reveals**

The computed advantages show:
- **Positive advantages**: Actions better than value function predicted
- **Negative advantages**: Actions worse than expected  
- **Magnitude**: Degree of surprise (larger = more unexpected)
- **Propagation**: How terminal reward influences earlier actions

## Applications in Modern RL

### PPO (Proximal Policy Optimization)
GAE is integral to PPO's advantage estimation:
```python
advantages = compute_advantages(rewards, values, gamma=0.99, lambda_=0.95)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### A3C (Asynchronous Advantage Actor-Critic)
Uses GAE for stable advantage estimation across multiple workers.

### TRPO (Trust Region Policy Optimization)
Relies on GAE for accurate policy gradient estimation.

## Implementation Variations

### Normalized Advantages
**Common Enhancement**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### Clipped Values
**Numerical Stability**
```python
advantages = np.clip(advantages, -clip_value, clip_value)
```

### Multiple Episodes
**Batch Processing**
```python
# Process multiple episodes in batch
for episode_rewards, episode_values in zip(batch_rewards, batch_values):
    episode_advantages = compute_advantages(episode_rewards, episode_values, gamma, lambda_)
```

## Key Insights

- **Bias-Variance Balance**: GAE provides tunable trade-off between estimation bias and variance
- **Efficient Computation**: Recursive formulation enables O(T) computation complexity
- **Universal Application**: Essential component in most modern policy gradient algorithms
- **Parameter Sensitivity**: λ parameter significantly affects learning dynamics

## Best Practices

- **λ Selection**: Start with λ = 0.95, tune based on environment characteristics
- **Value Function Quality**: Better value functions lead to better advantage estimates  
- **Normalization**: Usually beneficial to normalize advantages within each batch
- **Episode Boundaries**: Careful handling of terminal states is crucial for correctness

This implementation provides the foundation for understanding and implementing state-of-the-art policy gradient methods in reinforcement learning.