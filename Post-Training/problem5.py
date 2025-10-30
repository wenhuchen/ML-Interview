"""
Problem 5: Refactor the Transformer with pre-norm attention blocks, dropout,
and Rotary Position Embeddings (RoPE). Add enhanced sampling (temperature,
top-k, top-p) and a training loop using AdamW, cosine LR scheduling, and
gradient clipping.
"""

from problem3 import *
from torch import nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    """Improved attention mechanism with pre-norm, dropout, and better scaling"""
    
    def __init__(self, dim: int, head: int, dropout: float = 0.1):
        super().__init__()
        self.head = head
        self.dim = dim
        self.head_dim = dim // head
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP with better architecture
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor = None):        
        # Pre-norm attention
        normed_x = self.norm1(x)
        attn_out = self._attention(normed_x, seq_mask=seq_mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm MLP
        normed_x = self.norm2(x)
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x
    
    def _attention(self, x: torch.Tensor, seq_mask: torch.Tensor = None):
        batch_size, length, dim = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        mask = torch.triu(torch.ones((length, length), device=x.device), diagonal=1).bool()
        if seq_mask is not None:
            mask = mask.unsqueeze(0) | seq_mask.unsqueeze(1)
        else:
            mask = mask.unsqueeze(0)
        # Mask: [None, seq_len, seq_len]
        scores = scores.masked_fill(mask.unsqueeze(1), -10000)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, length, dim)
        
        return self.out_proj(out)


class Transformer(nn.Module):
    """Improved transformer with better architecture and training features"""
    
    def __init__(self, vocab: int, dim: int, head: int, layers: int, max_length: int = 20, 
                 dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.max_length = max_length
        self.eos = vocab - 1
        self.use_rope = use_rope
        
        # Token embedding
        self.embed = nn.Embedding(vocab, dim)
        
        # Positional encoding
        if use_rope:
            self._init_rope()
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, max_length, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            Attention(dim, head, dropout) for _ in range(layers)
        ])
        
        # Final layer norm and projection
        self.final_norm = nn.LayerNorm(dim)
        self.vocab_proj = nn.Linear(dim, vocab)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_rope(self):
        """Initialize Rotary Position Embedding (RoPE)"""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int):
        """Apply Rotary Position Embedding"""
        if not self.use_rope:
            return x + self.pos_embed[:, :seq_len]
        
        # Create position indices
        pos = torch.arange(seq_len, device=x.device).float()
        
        # Compute frequencies
        freqs = torch.outer(pos, self.inv_freq)
        
        # Create cos and sin matrices
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Apply rotation to even and odd dimensions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # Rotate
        x_rotated_even = x_even * cos.unsqueeze(0) - x_odd * sin.unsqueeze(0)
        x_rotated_odd = x_even * sin.unsqueeze(0) + x_odd * cos.unsqueeze(0)
        
        # Interleave back
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated
    
    def forward(self, x: torch.Tensor, project: bool = True):
        _, length = x.shape
        seq_mask = (x == self.eos)

        # Token embedding
        x = self.embed(x)
        
        # Apply positional encoding
        x = self._apply_rope(x, length)
        
        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, seq_mask=seq_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        if project:
            # Project to vocabulary
            return self.vocab_proj(x)
        else:
            return x
    
    def sample(self, x: torch.Tensor, max_new_tokens: int = None, 
               temperature: float = 1.0, top_k: int = None, top_p: float = None,
               sampling: bool = True):
        """Enhanced sampling with temperature, top-k, and top-p filtering"""
        if max_new_tokens is None:
            max_new_tokens = self.max_length - x.shape[1]
        
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = self.forward(x)[:, -1, :]
            
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
            x = torch.cat([x, next_token], dim=1)
            
            # Stop if EOS token is generated
            if (next_token == self.eos).all():
                break
        
        return x
    
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


def train_with_scheduler(model, steps: int = 1000, batch_size: int = 16, 
                        length: int = 12, vocab: int = 500, lr: float = 2e-3):
    """Training function with learning rate scheduling and gradient clipping"""
    loss_func = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        
        # Generate training data
        inputs = generate_train_sequence(batch_size, length, vocab)
        targets = torch.cat([
            inputs[:, 1:], 
            torch.ones((batch_size, 1), dtype=torch.int32, device=inputs.device) * model.eos
        ], -1)
        
        # Forward pass
        output = model(inputs)
        loss = loss_func(output.view(-1, output.shape[-1]), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            print(f'Step: {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')


if __name__ == '__main__':
    # Model parameters
    hidden_dim = 64  # Increased from 32
    vocab = 500
    num_heads = 4    # Increased from 2
    num_layers = 6   # Increased from 5
    max_length = 20
    
    # Create improved model
    model = Transformer(
        vocab=vocab, 
        dim=hidden_dim, 
        head=num_heads, 
        layers=num_layers,
        max_length=max_length,
        dropout=0.1,
        use_rope=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training parameters
    batch_size = 16
    length = 12
    steps = 1000
    
    # Train the model
    train_with_scheduler(model, steps, batch_size, length, vocab)
    
    # Test sampling
    model.eval()
    with torch.no_grad():
        # Test greedy sampling
        x = torch.randint(0, vocab, (batch_size, 1))
        y_greedy = model.sample(x, sampling=False)
        print("Greedy sampling results:")
        print(model.cutoff(y_greedy))
        
        # Test temperature sampling
        y_temp = model.sample(x, temperature=0.8, sampling=True)
        print("\nTemperature sampling results:")
        print(model.cutoff(y_temp))
        
        # Test top-k sampling
        y_topk = model.sample(x, temperature=1.0, top_k=10, sampling=True)
        print("\nTop-k sampling results:")
        print(model.cutoff(y_topk))
        
        # Test top-p sampling
        y_topp = model.sample(x, temperature=1.0, top_p=0.9, sampling=True)
        print("\nTop-p sampling results:")
        print(model.cutoff(y_topp))
