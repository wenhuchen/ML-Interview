"""
Problem 5: Add rotary/sinusoidal-style positional encoding to a causal
Transformer and implement both greedy and temperature-based stochastic sampling
for next-token generation on a toy sequence task.
"""

from problem4 import Attention, generate_train_sequence
from torch import nn
import math
import torch
import torch.nn.functional as F

class Transformer(nn.Module):

	def __init__(self, vocab: int, dim: int, head: int, layers: int, max_length: int = 20):
		super().__init__()
		self.vocab = vocab
		self.max_length = 10000
		inv_freq = (1.0 / 10000) ** (torch.arange(0, dim, 2).float() / dim)

		pos = torch.arange(self.max_length).float()
		angles = torch.einsum('i,j->ij', pos, inv_freq)
		self.cos_matrix = torch.cos(angles)
		self.sin_matrix = torch.sin(angles)

		self.embed = nn.Embedding(vocab + 1, dim)
		self.attention_layers = nn.ModuleList([Attention(dim, head) for _ in range(layers )])
		self.vocab_proj = nn.Linear(dim, vocab + 1)
		self.max_length = max_length
		self.eos = vocab

	def forward(self, x: torch.Tensor):
		batch_size, length = x.shape

		embeded_inputs = self.embed(x)
		embeded_inputs_odd = embeded_inputs[:, :, 0::2]
		embeded_inputs_even = embeded_inputs[:, :, 1::2]

		odd_cos = embeded_inputs_odd * self.cos_matrix[None, :length, :]
		even_sin = embeded_inputs_even * self.sin_matrix[None, :length, :]

		odd = embeded_inputs_odd * odd_cos + embeded_inputs_even * even_sin
		even = embeded_inputs_odd * odd_cos - embeded_inputs_even * even_sin

		combined = torch.stack([odd, even], dim=-1)
		combined = combined.reshape((batch_size, length, -1))

		x = combined
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
	optimizer = torch.optim.Adam(params = model.parameters(), lr=2e-3)

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
