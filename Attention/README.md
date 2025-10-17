# Attention Mechanisms and Visualization

This directory contains code for understanding and visualizing attention mechanisms in transformer models, specifically focusing on the LLaMA architecture.

## Overview

Attention is the core mechanism that allows transformer models to selectively focus on different parts of the input sequence. This collection provides tools to visualize and analyze attention patterns, along with a complete LLaMA implementation for experimentation.

## Files

### `attention_sink.py`
**Attention Pattern Visualization Script**

This script demonstrates how to extract and visualize attention patterns from a pre-trained LLaMA model:

- Loads Meta LLaMA-3-8B-Instruct model
- Processes a conversational input with system and user messages
- Extracts attention matrices from all transformer layers
- Generates visualizations for:
  - **Attention heatmaps**: Show which tokens attend to which other tokens
  - **Hidden state norms**: Track activation magnitudes across sequence positions
  - **Key-Value cache norms**: Analyze the stored attention key and value representations

**Key Features:**
- Multi-layer attention analysis (first 10 layers)
- Token-level attention pattern visualization 
- Norm analysis for hidden states and KV cache
- Saves PNG plots for each layer

### `modeling_llama.py`
**Complete LLaMA Model Implementation**

A comprehensive implementation of the LLaMA architecture with attention mechanism details:

**Core Components:**
- **`LlamaAttention`**: Multi-head attention with RoPE (Rotary Position Embedding)
- **`LlamaFlashAttention2`**: Memory-efficient attention using FlashAttention
- **`LlamaSdpaAttention`**: PyTorch's scaled dot-product attention implementation
- **`LlamaRotaryEmbedding`**: Rotary positional encoding for better length extrapolation
- **`LlamaMLP`**: SwiGLU feed-forward network
- **`LlamaDecoderLayer`**: Complete transformer block with pre-normalization

**Advanced Features:**
- Multiple attention implementations (eager, flash_attention_2, sdpa)
- Dynamic RoPE scaling for longer sequences
- KV caching for efficient generation
- Gradient checkpointing support
- Mixed precision training compatibility

## Attention Mechanism Concepts

### Multi-Head Attention
The attention mechanism computes three matrices from input:
- **Query (Q)**: What information to look for
- **Key (K)**: What information is available  
- **Value (V)**: The actual information content

The attention score is computed as: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

### Rotary Position Embedding (RoPE)
- Encodes positional information directly into query and key vectors
- Enables better length extrapolation compared to absolute position embeddings
- Applies rotation transformations based on position indices

### Key-Value Caching
- Stores computed key and value matrices from previous tokens
- Enables efficient autoregressive generation without recomputing past attention
- Critical for inference performance in language models

### Grouped Query Attention (GQA)
- LLaMA uses fewer key-value heads than query heads for efficiency
- Reduces memory usage while maintaining model quality
- `num_key_value_groups = num_heads // num_key_value_heads`

## Understanding Attention Patterns

The visualization script helps analyze:

1. **Attention Sinks**: Some tokens (often the first token) receive disproportionate attention
2. **Local vs Global Attention**: How attention spreads across different sequence positions  
3. **Layer-wise Evolution**: How attention patterns change through the model depth
4. **Head Specialization**: Different attention heads may focus on different linguistic patterns

## Running the Code

### Attention Visualization:
```bash
python attention_sink.py
```
**Requirements:**
- Hugging Face Transformers library
- Access to Meta LLaMA-3-8B-Instruct model
- Sufficient GPU memory (model is loaded in bfloat16)

**Output:**
- `layer_{i}_attention.png`: Attention heatmaps for each layer
- `hidden_states_{i}_norm.png`: Hidden state magnitude plots
- `kv_{i}_norm.png`: Key-value cache norm analysis

### Model Usage:
The `modeling_llama.py` can be imported and used as a drop-in replacement for the standard LLaMA implementation with enhanced attention visualization capabilities.

## Key Insights

- **Attention Visualization**: Reveals how models process and focus on different parts of input
- **Layer Analysis**: Early layers often capture syntactic patterns, later layers semantic relationships
- **Performance Optimization**: Different attention implementations (Flash, SDPA) provide memory/speed tradeoffs
- **Position Encoding**: RoPE enables models to handle sequences longer than training data

This directory provides both theoretical understanding and practical tools for analyzing attention mechanisms in state-of-the-art language models.