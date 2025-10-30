"""
Hugging Face Transformer Integration for RL Training

This module integrates Hugging Face's GPT-2 model with the RL training pipeline
from problem9.py. It reuses all utility functions from problem9.py while providing
a more robust transformer implementation using the official Hugging Face library.

Key improvements:
- Uses production-ready GPT-2 implementation
- Better memory efficiency and performance
- Advanced sampling strategies (temperature, top-k, top-p)
- Maintains full compatibility with existing RL training loop
"""

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import numpy as np
from problem9 import train_data_generator, compute_rewards, save_model, load_model


class HuggingFaceTransformer(nn.Module):
    """Wrapper for Hugging Face GPT-2 model to maintain compatibility with existing training loop"""
    
    def __init__(self, vocab: int, dim: int, head: int, layers: int, max_length: int = 20):
        super().__init__()
        self.vocab = vocab
        self.max_length = max_length
        self.eos = vocab
        
        # Create GPT-2 configuration
        config = GPT2Config(
            vocab_size=vocab,  # +1 for EOS token
            n_positions=max_length,
            n_ctx=max_length,
            n_embd=dim,
            n_layer=layers,
            n_head=head,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=False,  # Disable caching for training
        )
        
        # Initialize the model
        self.model = GPT2LMHeadModel(config)
        
        # Create a simple tokenizer-like mapping for our integer tokens
        self.token_to_id = {i: i for i in range(vocab)}
        self.token_to_id['<|endoftext|>'] = vocab - 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
    
    def forward(self, input_ids: torch.Tensor):
        """Forward pass through the model"""
        outputs = self.model(input_ids=input_ids)
        return outputs.logits
    
    def sample(self, input_ids: torch.Tensor, max_new_tokens: int = None, 
               temperature: float = 1.0, top_k: int = None, top_p: float = None,
               sampling: bool = True):
        """Sample from the model with various strategies"""
        if max_new_tokens is None:
            max_new_tokens = self.max_length - input_ids.shape[1]
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for the last token
            with torch.no_grad():
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                topk_logits, topk_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, topk_indices, topk_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample next token
            if sampling:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if (next_token == self.eos).all():
                break
        
        return generated
    
    def cutoff(self, t: torch.Tensor):
        """Cut sequences at EOS tokens"""
        batch_size, length = t.shape
        mask = (t == self.eos)
        indices = torch.arange(length, device=t.device).expand(batch_size, -1)
        indices = torch.where(mask, indices, length)
        first_eos_idx = torch.min(indices, dim=1).values
        
        # Output as list of lists (ragged)
        result = [
            t[i, :first_eos_idx[i]].tolist() if first_eos_idx[i] < length else t[i].tolist()
            for i in range(batch_size)
        ]
        return result


# All utility functions are imported from problem9.py


def train_with_huggingface_transformer():
    """Main training function using Hugging Face GPT-2 model"""
    print("Training with Hugging Face GPT-2 Transformer...")
    
    # Model parameters
    hidden_dim = 64
    vocab = 30
    batch_size = 32
    length = 12
    steps = 400
    group_size = 50

    # Create Hugging Face transformer model
    model = HuggingFaceTransformer(
        vocab=vocab, 
        dim=hidden_dim, 
        head=8, 
        layers=12, 
        max_length=length + 2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    optimizer.zero_grad()

    # init_step = load_model(model, optimizer, 'checkpoint_199.pt')
    init_step = 0
    
    for step in range(0, steps):
        optimizer.zero_grad()
        inputs, reference = train_data_generator(batch_size, vocab, length)
        rollouts = []
        
        # Use temperature scaling for better exploration
        temperature = max(0.8, 1.0 - step / steps)  # Start high, decrease over time
        
        for i in range(group_size):
            rollout = model.sample(inputs, sampling=True, temperature=temperature)
            rollouts.append(rollout)
        rollouts = torch.stack(rollouts, 1)

        # rewards: [batch, group]
        unnormalized_rewards, normalized_rewards = compute_rewards(rollouts, reference)

        logits = model(rollouts.view(-1, rollouts.shape[-1])[:, :-1])
        logits = logits[:, inputs.shape[1]-1:, :]
        neg_log_probs = -F.log_softmax(logits, -1)

        targets = rollouts[:, :, inputs.shape[1]:]
        targets = targets.reshape(-1, targets.shape[-1])

        # Get log probabilities for the target tokens
        loss = torch.gather(neg_log_probs, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Reshape to [batch_size, group_size, sequence_length]
        loss = loss.view(batch_size, group_size, -1)
        
        # Calculate entropy bonus to encourage exploration
        probs = F.softmax(logits.view(batch_size, group_size, -1, logits.shape[-1]), dim=-1)
        entropy = -(probs * F.log_softmax(logits.view(batch_size, group_size, -1, logits.shape[-1]), dim=-1)).sum(dim=-1)
        entropy_bonus = 0.01 * entropy.mean()  # Small entropy bonus
        
        # Apply rewards to the loss (policy gradient)
        # rewards: [batch_size, group_size] -> expand to match loss shape
        rl_loss = loss * normalized_rewards.unsqueeze(-1)  # [batch_size, group_size, seq_len]
        
        # Sum over sequence length, then take mean over group_size, then sum over batch
        rl_loss = rl_loss.sum(dim=-1).mean(dim=1).sum() - entropy_bonus

        rl_loss.backward()
        optimizer.step()
        
        print(f'step: {step}, loss: {rl_loss.item():.4f}, reward: {unnormalized_rewards.mean().item():.4f}')

        if step % 10 == 0 and step > 0:  # More frequent evaluation
            input, reference = train_data_generator(batch_size=batch_size, vocab=vocab, length=length)
            y = model.sample(input)
            success = 0
            failure = 0
            close_matches = 0

            for i, (y_, reference_) in enumerate(zip(y, reference)):
                rollout_text = str(y_.tolist())[1:-1]
                groundtruth = str(reference_.tolist())[1:-1]
                target_avg = reference_[0].item()
                
                if groundtruth in rollout_text:
                    success += 1
                else:
                    # Check for close matches
                    rollout_tokens = rollout_text.split(', ')
                    for token_str in rollout_tokens:
                        try:
                            token_val = int(token_str.strip())
                            if abs(token_val - target_avg) <= 1:
                                close_matches += 1
                                break
                        except ValueError:
                            continue
                    failure += 1
            
            total = success + failure
            print(f'step: {step}, success: {success/total:.3f}, close: {close_matches/total:.3f}, avg reward: {unnormalized_rewards.mean().item():.3f}')
            
            # Show example for debugging
            if step % 50 == 0:
                print(f"Example - Input: {input[0].tolist()}, Target avg: {reference[0][0].item()}, Generated: {y[0].tolist()}")

    save_model(model, optimizer, step)
    print("Training completed!")


if __name__ == '__main__':
    print("Hugging Face Transformer Integration")
    print("Reusing functions from problem9.py for RL training")
    print("=" * 60)
    
    # Run the main training using Hugging Face GPT-2
    # All utility functions (train_data_generator, compute_rewards, etc.) 
    # are imported from problem9.py
    train_with_huggingface_transformer()
