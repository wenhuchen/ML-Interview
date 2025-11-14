"""
Mixture-of-Experts (MoE) feed-forward for GPT.

Adds top-k gated MoE layers to the sliding-window GPT, with KV caching and
simple generation utilities for demonstration.
"""

import time
import tiktoken
import torch
import torch.nn as nn
from problem16 import MultiHeadAttention, LayerNorm, GPTModel


class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]

        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)
        self.fc1 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor):
        scores = self.gate(x)

        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, -1)
        topk_prob = torch.softmax(topk_scores, -1)
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, -1)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_prob.reshape(-1, self.num_experts_per_tok)
        out_flat = torch.zeros(batch_size * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        unique_experts = torch.unique(topk_indices_flat)

        for exper_id in unique_experts:
            exper_id = exper_id.item()

            # [batch x seq, num_experts], it is a binary mask
            mask = topk_indices_flat == exper_id
            if not mask.any():
                break

            # [batch x seq, True/False]
            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            # [selected, dim]
            expert_input = x_flat.index_select(0, selected_idx)
            hidden = torch.nn.functional.silu(self.fc1[exper_id](expert_input)) * self.fc2[exper_id](expert_input)
            expert_output = self.fc3[exper_id](hidden)

            print('batch_size: ', batch_size, 'length: ', seq_len, 'selected: ', selected_idx.shape)
            # [selected, num_experts]
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)

            # [selected, 1]
            selected_probs = torch.gather(
                input=topk_probs_flat.index_select(0, selected_idx), 
                dim=-1, 
                index=slot_indices
            ).squeeze(-1)

            out_flat.index_add(0, selected_idx, expert_output * selected_probs.unsqueeze(-1))

        return out_flat.reshape(batch_size, seq_len, self.emb_dim)


class TransformerMoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            emb_size=cfg['emb_dim'],
            num_head=cfg['n_heads'],
            sliding_window=cfg['sliding_window']
        )
        self.mlp = MoEFeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.cache_key = None
        self.cache_value = None

    def forward(self, inputs: torch.Tensor, use_cache=False):
        output, headed_key, headed_value = self.attention(self.norm1(inputs), self.cache_key, self.cache_value)
        if use_cache:
            self.cache_key = headed_key
            self.cache_value = headed_value

        output = inputs + output
        output = output + self.mlp(self.norm2(output))
        return output

    def reset_kv_cache(self):
        self.cache_key = None
        self.cache_value = None


class GPTMoEModel(GPTModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.transformer_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerMoEBlock(cfg) for _ in range(cfg['n_layers'])]
        )


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.position_embedding.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def encode_batch_left_pad(texts, tokenizer, max_length=None, pad_token_id=50256):
    """Left padding for efficient generation"""
    encoded_texts = [tokenizer.encode(text) for text in texts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_length is None:
        max_length = max(len(seq) for seq in encoded_texts)
    batch_size = len(texts)
    batch_tensor = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    for i, seq in enumerate(encoded_texts):
        seq_len = len(seq)
        start_idx = max_length - seq_len  # Start from the right
        batch_tensor[i, start_idx:] = torch.tensor(seq, device=device)
    return batch_tensor


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 1,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "sliding_window": 16,
        "num_experts_per_tok": 3,
        "num_experts": 16,
        "hidden_dim": 3072,
    }

    torch.manual_seed(123)
    model = GPTMoEModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    contexts = ["Hello, I am", "The quick brown fox", "To be or not to be"]

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_tensor = encode_batch_left_pad(contexts, tokenizer)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("Sliding window: ", GPT_CONFIG_124M['sliding_window'])
    print("\nInput text:", contexts)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    ####################################################
    # NEW
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids[0].tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
