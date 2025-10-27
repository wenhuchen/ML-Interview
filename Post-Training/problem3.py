"""
Problem 4: Build a causal Transformer with standard multi-head attention and a
multi-query attention (MQA) variant. Train on a synthetic sequence-continuation
task and provide sampling plus EOS-based cutoff utilities.
"""

import torch
from torch import nn
from math import sqrt
import random

class Attention(nn.Module):

	def __init__(self, dim: int, head: int):
		super().__init__()
		self.head = head
		self.dim = dim
		self.q_proj = nn.Linear(dim, dim)
		self.k_proj = nn.Linear(dim, dim)
		self.v_proj = nn.Linear(dim, dim)

		self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
		self.layer_norm = nn.LayerNorm(dim)

	def post_layernorm(self, x, f):
		return self.layer_norm(x + f(x))

	def attention(self, x):
		batch_size, length, dim = x.shape
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)
		q_headed = q.view(batch_size, length, self.head, dim // self.head)
		k_headed = k.view(batch_size, length, self.head, dim // self.head)
		v_headed = v.view(batch_size, length, self.head, dim // self.head)
		pre_attention = torch.einsum('blhd,bmhd->bhlm', q_headed, k_headed) / sqrt(self.dim // self.head)
		mask = torch.triu(torch.ones((length, length)), diagonal=1).bool()
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


class MQA_Attention(Attention):

	def __init__(self, dim: int, head: int):
		super().__init__(dim, head)
		self.head = head
		self.dim = dim
		self.q_proj = nn.Linear(dim, dim)
		self.k_proj = nn.Linear(dim, dim // self.head)
		self.v_proj = nn.Linear(dim, dim // self.head)

		self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
		self.layer_norm = nn.LayerNorm(dim)

	def attention(self, x):
		batch_size, length, dim = x.shape
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		q_headed = q.view(batch_size, length, self.head, dim // self.head)

		pre_attention = torch.einsum('blhd,bmd->bhlm', q_headed, k) / sqrt(self.dim // self.head)
		mask = torch.triu(torch.ones((length, length)), diagonal=1).bool()
		pre_attention = pre_attention.masked_fill(mask, -1000)
		attention = nn.Softmax(-1)(pre_attention)
		
		output = torch.einsum('bhlm,bmd->blhd', attention, v)
		output = output.reshape((output.shape[0], output.shape[1], -1))
		return output


class Transformer(nn.Module):

	def __init__(self, vocab: int, dim: int, head: int, layers: int, attention_type: str):
		super().__init__()
		self.vocab = vocab
		self.embed = nn.Embedding(vocab + 1, hidden_dim)
		if attention_type == 'standard':
			self.attention_layers = nn.ModuleList([Attention(dim, head) for _ in range(layers )])
		else:
			self.attention_layers = nn.ModuleList([MQA_Attention(dim, head) for _ in range(layers )])
		self.vocab_proj = nn.Linear(dim, vocab + 1)
		self.max_length = 20
		self.pos_embed = nn.Parameter(torch.randn(1, self.max_length, dim))
		self.eos = vocab

	def forward(self, x: torch.Tensor):
		batch_size, length = x.shape
		embeded_inputs = self.embed(x)
		embeded_inputs = embeded_inputs + self.pos_embed[:, :embeded_inputs.size(1)]
		x = embeded_inputs
		for layer in self.attention_layers:
			x = layer(x)

		x = self.vocab_proj(x.view(-1, x.shape[-1]))
		logits = x.view((batch_size, length, self.vocab + 1))
		return logits

	def sample(self, x: torch.Tensor):
		for _ in range(x.shape[1], self.max_length):
			logits = self.forward(x)[:, -1, :]
			idx = torch.argmax(logits, dim=-1)
			x = torch.cat([x, idx.unsqueeze(-1)], 1)
		assert x.shape[-1] == self.max_length
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


def generate_train_sequence(batch_size: int, length: int, vocab: int):
	sequence = []
	for i in range(batch_size):
		x = random.randint(0, vocab)
		y = [_ % vocab for _ in range(x, x + length)]
		sequence.append(y)
	sequence = torch.asarray(sequence)
	return sequence


if __name__  == '__main__':
	hidden_dim = 32
	vocab = 500
	model = Transformer(vocab=vocab, dim=hidden_dim, head=2, layers=5, attention_type='MQA')
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
