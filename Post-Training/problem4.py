"""
Problem 4: Add rotary/sinusoidal-style positional encoding to a causal
Transformer and implement both greedy and temperature-based stochastic sampling
for next-token generation on a toy sequence task.
"""

from torch import nn
import torch
import torch.nn.functional as F
from problem3 import generate_train_sequence
from math import sqrt

class Attention(nn.Module):

	def __init__(self, dim: int, head: int):
		super().__init__()
		self.head = head
		self.dim = dim
		self.head_dim = dim // head
		self.q_proj = nn.Linear(dim, dim)
		self.k_proj = nn.Linear(dim, dim)
		self.v_proj = nn.Linear(dim, dim)

		self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
		self.layer_norm = nn.LayerNorm(dim)
		self.max_length = 10000
		
		# RoPE initialization - compute for head_dim, not full dim
		inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
		pos = torch.arange(self.max_length).float()
		angles = torch.einsum('i,j->ij', pos, inv_freq)
		
		# Register as buffers so they move to the correct device
		self.register_buffer('cos_matrix', torch.cos(angles))
		self.register_buffer('sin_matrix', torch.sin(angles))

	def post_layernorm(self, x, f):
		return self.layer_norm(x + f(x))

	def _apply_rope(self, x: torch.Tensor):
		"""
		Apply Rotary Position Embedding to query or key tensors.
		x shape: (batch_size, num_heads, seq_len, head_dim)
		"""
		batch_size, num_heads, seq_len, head_dim = x.shape
		
		# Get cos and sin for current sequence length
		cos = self.cos_matrix[:seq_len, :]  # (seq_len, head_dim // 2)
		sin = self.sin_matrix[:seq_len, :]  # (seq_len, head_dim // 2)
		
		# Split into even and odd dimensions
		x1 = x[..., 0::2]  # Even dimensions
		x2 = x[..., 1::2]  # Odd dimensions
		
		# Broadcast cos/sin: (seq_len, head_dim//2) -> (1, 1, seq_len, head_dim//2)
		cos = cos.unsqueeze(0).unsqueeze(0)
		sin = sin.unsqueeze(0).unsqueeze(0)
		
		# Apply rotation
		x1_rotated = x1 * cos - x2 * sin
		x2_rotated = x1 * sin + x2 * cos
		
		# Interleave back together
		x_rotated = torch.zeros_like(x)
		x_rotated[..., 0::2] = x1_rotated
		x_rotated[..., 1::2] = x2_rotated
		
		return x_rotated

	def attention(self, x):
		batch_size, length, dim = x.shape
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		# Reshape to (batch, length, num_heads, head_dim) then transpose to (batch, num_heads, length, head_dim)
		q_headed = q.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)
		k_headed = k.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)
		v_headed = v.view(batch_size, length, self.head, self.head_dim).transpose(1, 2)

		# Apply RoPE to queries and keys (NOT to values!)
		q_headed = self._apply_rope(q_headed)
		k_headed = self._apply_rope(k_headed)

		# Transpose back for einsum: (batch, num_heads, length, head_dim)
		q_headed = q_headed.transpose(1, 2)
		k_headed = k_headed.transpose(1, 2)
		v_headed = v_headed.transpose(1, 2)

		pre_attention = torch.einsum('blhd,bmhd->bhlm', q_headed, k_headed) / sqrt(self.head_dim)
		mask = torch.triu(torch.ones((length, length), device=x.device), diagonal=1).bool()
		pre_attention = pre_attention.masked_fill(mask, -1000)
		attention = nn.Softmax(-1)(pre_attention)

		output = torch.einsum('bhlm,bmhd->blhd', attention, v_headed)
		output = output.reshape((output.shape[0], output.shape[1], -1))
		return output

	def forward(self, x: torch.Tensor):
		batch_size, length, dim = x.shape
		output = self.post_layernorm(x, self.attention)
		output = self.post_layernorm(output, self.mlp)
		return output


class Transformer(nn.Module):

	def __init__(self, vocab: int, dim: int, head: int, layers: int, max_length: int = 20):
		super().__init__()
		self.vocab = vocab

		self.embed = nn.Embedding(vocab + 1, dim)
		self.attention_layers = nn.ModuleList([Attention(dim, head) for _ in range(layers )])
		self.vocab_proj = nn.Linear(dim, vocab + 1)
		self.max_length = max_length
		self.eos = vocab

	def forward(self, x: torch.Tensor):
		batch_size, length = x.shape

		embeded_inputs = self.embed(x)
		x = embeded_inputs

		for layer in self.attention_layers:
			x = layer(x)

		x = self.vocab_proj(x.view(-1, x.shape[-1]))
		logits = x.view((batch_size, length, self.vocab + 1))
		return logits

	def sample(self, x: torch.Tensor, sampling=False, temperature=1.0):
		if not sampling:
			for _ in range(x.shape[1], self.max_length):
				logits = self.forward(x)[:, -1, :]
				idx = torch.argmax(logits, dim=-1)
				x = torch.cat([x, idx.unsqueeze(-1)], 1)
		else:
			for _ in range(x.shape[1], self.max_length):
				logits = self.forward(x)[:, -1, :]
				# Apply temperature scaling
				logits = logits / temperature
				prob = F.softmax(logits, -1)
				idx = torch.multinomial(prob, num_samples=1)
				x = torch.cat([x, idx], 1)
		return x

	def cutoff(self, t: torch.Tensor):
		batch_size, length = t.shape
		mask = (t == self.eos)
		indices = torch.arange(length).expand(batch_size, -1)
		indices = torch.where(mask, indices, length)
		first_zero_idx = torch.min(indices, dim=1).values  # index of first zero per row

		# Output as list of lists (ragged)
		result = [
		    t[i, :first_zero_idx[i]].tolist() if first_zero_idx[i] < length else t[i].tolist()
		    for i in range(batch_size)
		]
		return result


if __name__  == '__main__':
	hidden_dim = 32
	vocab = 500
	model = Transformer(vocab=vocab, dim=hidden_dim, head=2, layers=5)
	batch_size = 16
	length = 12
	steps = 1000

	loss_func = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3)

	for step in range(steps):
		optimizer.zero_grad()
		inputs = generate_train_sequence(batch_size, length, vocab)
		targets = torch.cat([inputs[:, 1:], torch.ones((batch_size, 1), dtype=torch.int32) * model.eos], -1)
		output = model(inputs)
		loss = loss_func(output.view((-1, output.shape[-1])), targets.view(-1))
		loss.backward()
		optimizer.step()
		print('Current step: ', step, 'Loss function: ', loss.item())

	x = torch.randint(0, vocab, (batch_size, 1))
	y = model.sample(x)
	print(model.cutoff(y))
