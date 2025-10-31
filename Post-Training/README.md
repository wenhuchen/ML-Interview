# Post-Training (ML Interview)

This repository collects a set of small, focused Python scripts exploring post-training techniques for language models and adjacent building blocks (tokenization, transformers, attention, retrieval, RLHF, DPO, etc.). Most scripts are self‑contained and can be run directly as demos.

## What’s Inside

- `problem1.py`: MinHash utilities for near‑duplicate detection. Implements an LSH‑style MinHash to estimate Jaccard similarity between document shingles for deduplication.
- `problem2.py`: Simple BPE tokenizer from scratch. Learns merges and encodes/decodes without external libraries.
- `problem3.py`: Causal Transformer backbone and utilities. Baseline multi‑head attention plus sampling helpers for a toy sequence task.
- `problem4.py`: Positional encoding and sampling. Adds rotary/sinusoidal‑style positions and greedy/temperature sampling.
- `problem5.py`: Refined Transformer with pre‑norm, dropout, RoPE, and advanced sampling (temperature, top‑k, top‑p), plus a training loop.
- `problem6.py`: Numpy backward for attention. Derives and checks gradients for scaled dot‑product attention and softmax.
- `problem7.py`: Memory‑efficient attention. Chunked keys/values with numerically stable softmax for long sequences.
- `problem8.py`: TF–IDF‑style retrieval demo. Builds sparse vectors with a simple BPE and ranks corpus by query similarity.
- `problem9.py`: RL fine‑tuning on a sequence task. Group rollouts, normalized substring rewards, entropy bonus, and checkpointing.
- `problem10.py`: Hugging Face GPT‑2 wrapper. Integrates an HF model into the same RL training pipeline.
- `problem11.py`: GRPO training script. Group Relative Policy Optimization with SFT warm‑up, clipping vs. a frozen reference, and evaluation.
- `problem12.py`: Reward model training. Small Transformer that scores positive/negative pairs with left‑padding batch encoding.
- `problem13.py`: RLHF training loop. PPO‑style optimization using the learned reward model and log‑prob utilities.
- `problem14.py`: RLHF with value head. Actor‑critic variant that adds a value network to the Transformer.
- `problem15.py`: From‑scratch GPT components. Collects Chapter 3–4 building blocks into a minimal GPT with KV cache.
- `problem16.py`: Sliding‑window GPT. Windowed masking over keys/values and cache‑aware generation for long contexts.
- `problem17.py`: MoE feed‑forward for GPT. Top‑k gated Mixture‑of‑Experts integrated into the sliding‑window model.
- `problem18.py`: Group testing simulation. Divide‑and‑conquer identification of defective nodes with complexity tracing.
- `problem19.py`: DPO training demo. Direct Preference Optimization against a frozen reference on toy preferences.
- `problem20.py`: Distributed Training with FSDP. Memory profiling demo showing how GPU memory consumption changes with different FSDP configurations.
- `problem21.py`: KL Divergence calculations. Explores different forms of KL divergence (forward, reverse, symmetric) and applications in RLHF, PPO, DPO, and GRPO.

## Running Examples

- Ensure Python 3.9+ and PyTorch installed where required. Some scripts optionally use CUDA if available.
- Run any script directly, e.g. `python problem5.py` or `python problem19.py`. Some demos print intermediate metrics or decoded samples.

## Notes

- Scripts are intentionally compact and pedagogical; many use small synthetic datasets and simplified objectives.
- Several files share helpers (e.g., tokenizer or Transformer). See imports at the top of each module for dependencies.

