import torch
import torch.nn as nn
import math


class RoPEEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Rotary Positional Embedding (RoPE)
        :param dim: The dimensionality of the model (embedding dimension).
        """
        super().__init__()
        self.dim = dim
        # Precompute frequencies for even dimensions
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, shift: int):
        """
        Apply RoPE to input embeddings.
        :param x: Tensor of shape (batch_size, head_dim, seq_len, dim)
        :return: Tensor of shape (batch_size, head_dim, seq_len, dim) with RoPE applied
        """
        _, _, seq_len, _ = x.shape
        pos = torch.arange(shift, seq_len + shift, device=x.device).unsqueeze(1)

        sin_cos = pos * self.inv_freq.unsqueeze(0)  # (seq_len, dim/2)
        sin, cos = sin_cos.sin(), sin_cos.cos()

        x_odd, x_even = torch.chunk(x, 2, dim=-1)

        x_transformed_odd = x_odd * cos[..., :, :] - x_even * sin[..., :, :]
        x_tranformed_even = x_even * cos[..., :, :] + x_odd * sin[..., :, :]
        y = torch.cat([x_transformed_odd, x_tranformed_even], -1)
        return y


class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Self-Attention with Rotary Positional Embedding.
        :param embed_dim: The dimensionality of the embeddings.
        :param num_heads: The number of attention heads.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RoPEEmbedding(self.head_dim)

    def forward(self, x):
        """
        Forward pass for self-attention with RoPE.
        :param x: Input tensor of shape (batch_size, seq_len, embed_dim).
        :return: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)

        outputs = []
        for shift in [16, 32]:
            # Apply RoPE to Q and K
            q_rotated = self.rope(
                q.permute(0, 2, 1, 3), shift=shift).permute(0, 2, 1, 3)
            k_rotated = self.rope(
                k.permute(0, 2, 1, 3), shift=shift).permute(0, 2, 1, 3)

            # Scaled dot-product attention
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = torch.einsum(
                "bqhd,bkhd->bhqk", q_rotated, k_rotated) / math.sqrt(self.head_dim)
            attn_weights = attn_weights.masked_fill(
                causal_mask == 0, float('-inf'))

            # Softmax and attention output
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v)

            # Combine heads and project
            attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

            # Store outputs in an array
            outputs.append([attn_probs, self.out_proj(attn_output)])

        return outputs


# Example Usage
if __name__ == "__main__":

    # Simple unit test;
    rope = RoPEEmbedding(8)
    inputs = torch.randn(1, 1, 5, 8)
    x = rope(inputs, shift=0).squeeze()
    print(x @ x.T)

    y = rope(inputs, shift=10).squeeze()
    print(y @ y.T)

    # Random case
    batch_size, seq_len, embed_dim, num_heads = 2, 1600, 256, 1
    X = torch.randn(batch_size, seq_len, embed_dim)
    net = SelfAttentionWithRoPE(embed_dim, num_heads)

    for dtype in [torch.bfloat16, torch.float32, torch.float64]:
        print(dtype, '--'*32)
        x = X.to(dtype)
        self_attention = net.to(dtype)
        with torch.autocast(device_type='cuda', dtype=dtype):
            with torch.no_grad():
                outputs = self_attention(x)
                print(outputs[0][0][0][0])
                print(outputs[1][0][0][0])
                print('Error: ', (outputs[0][0][0][0] - outputs[1][0][0][0]).abs().sum())