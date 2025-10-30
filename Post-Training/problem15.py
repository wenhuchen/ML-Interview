"""
From-scratch GPT building blocks with KV cache.

Consolidates Chapter 3–4 components (attention, layer norm, MLP, blocks)
into a minimal GPT model and demo code runnable as a script.
"""

import time
import tiktoken
import torch
import torch.nn as nn
import torch
import math


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
	def __init__(self, emb_size: int, num_head: int):
		super().__init__()
		self.k_proj = nn.Linear(emb_size, emb_size)
		self.q_proj = nn.Linear(emb_size, emb_size)
		self.v_proj = nn.Linear(emb_size, emb_size)
		self.emb_size = emb_size
		self.num_head = num_head
		self.softmax = nn.Softmax(-1)

	def forward(self, q: torch.Tensor, cache_key: torch.Tensor = None, cache_value: torch.Tensor = None):
		query = self.q_proj(q)
		headed_query = query.view(query.shape[0], query.shape[1], self.num_head, self.emb_size // self.num_head)

		key = self.k_proj(q)
		value = self.v_proj(q)
		
		headed_key = key.view(key.shape[0], key.shape[1], self.num_head, self.emb_size // self.num_head)
		headed_value = value.view(key.shape[0], key.shape[1], self.num_head, self.emb_size // self.num_head)

		if cache_key is not None and cache_value is not None:
			headed_key = torch.cat([cache_key, headed_key], 1)
			headed_value = torch.cat([cache_value, headed_value], 1)
			mask = torch.cat([
				torch.zeros(q.shape[1], cache_key.shape[1]), 
				torch.triu(torch.ones(q.shape[1], q.shape[1]), diagonal=1)
			], -1)
		else:
			mask = torch.triu(torch.ones((q.shape[1], q.shape[1])), diagonal=1)

		attention_score = torch.einsum('bqhd,bkhd->bhqk', headed_query, headed_key) / math.sqrt(self.num_head // self.num_head)
		attention_score.masked_fill(mask, -math.inf)

		attention_weights = self.softmax(attention_score)

		output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, headed_value)
		output = output.reshape((output.shape[0], output.shape[1], -1))

		return output, headed_key, headed_value


class LayerNorm(nn.Module):
	def __init__(self, emb_size: int):
		super().__init__()
		self.scale = nn.Parameter(torch.ones(emb_size))
		self.shift = nn.Parameter(torch.zeros(emb_size)) 

	def forward(self, x: torch.Tensor):
		x = (x -  x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-8)
		return self.scale * x + self.shift



class GELU(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor):
		return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
	def __init__(self, emb_size: int):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(emb_size, 4 * emb_size),
			GELU(),
			nn.Linear(4 * emb_size, emb_size)
		)

	def forward(self, x: torch.Tensor):
		return self.mlp(x)


class TransformerBlock(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.attention = MultiHeadAttention(
			emb_size=cfg['emb_dim'],
			num_head=cfg['n_heads']
		)
		self.mlp = FeedForward(
			emb_size=cfg['emb_dim']
		)
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


class GPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
		)
		self.embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
		self.position_embedding = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
		self.norm = LayerNorm(cfg['emb_dim'])
		self.vocab_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])
		self.current_length = 0

	def forward(self, x: torch.Tensor, use_cache=False):
		embedding = self.embedding(x)
		if use_cache:
			positions = torch.arange(self.current_length, self.current_length + x.shape[-1], dtype=torch.long)
			self.current_length += x.shape[-1]
		else:
			positions = torch.arange(x.shape[-1], dtype=torch.long)
		pos_embedding = self.position_embedding(positions)
		embedding = embedding + pos_embedding[None, :, :]
		output = embedding
		for layer in self.transformer_blocks:
			output = layer(output, use_cache)
		output = self.norm(output)
		logits = self.vocab_proj(output)
		return logits

	def reset_kv_cache(self):
		self.current_length = 0
		for block in self.transformer_blocks:
			block.reset_kv_cache()


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


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    ####################################################
    # NEW
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=40,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

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
