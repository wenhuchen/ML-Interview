"""
Reward model for toy preference data.

Trains a small Transformer to score texts from simple positive/negative
pairs and includes a left-padding batch encoding utility.
"""

from problem2 import BPETokenizer
from itertools import chain
from problem5 import Transformer
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from torch import optim

pairs = [
	{"positive": "You are good", "negative": "You are bad"},
	{"positive": "I am good", "negative": "I am bad"},
	{"positive": "Your performance is good", "negative": "Your performance is bad"},
	{"positive": "Your money is good", "negative": "Your money is bad"},
	{"positive": "Your health is good", "negative": "Your health is bad"},
	{"positive": "Health is good", "negative": "Your health is bad"},
	{"positive": "White is good", "negative": "Black is bad"},
	{"positive": "White is good", "negative": "White is bad"},
	{"positive": "Black is good", "negative": "White is bad"},
	{"positive": "Black is good", "negative": "Black is bad"}
]

eval_text = "Black are good"
# eval_text = "Black are bad"

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


class RewardModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.model = Transformer(
	        vocab=config['vocab'], 
	        dim=config['hidden_dim'], 
	        head=config['head'], 
	        layers=config['layers'],
	        max_length=config['max_length'],
	        dropout=0.1,
	        use_rope=True
    	)
		self.reward_proj = nn.Linear(config['vocab'], 1)

	def forward(self, x: torch.Tensor):
		logits = self.model(x)
		reward = self.reward_proj(logits[:, -1, :])
		return reward.squeeze(-1)


if __name__ == "__main__":
	tokenizer = BPETokenizer(vocab_size=100)

	corpus = chain(*[(_['positive'], _['negative'])for _ in pairs])
	tokenizer.train(corpus)
	tokenizer.expand_vocab(['<pad>'])
	print(tokenizer.vocab)

	config = {
		'hidden_dim': 64,
		'vocab': len(tokenizer.vocab),
		'batch_size': 32,
		'length': 12,
		'sft_steps': 50,
		'steps': 400,
		'group_size': 50,
		'head': 8,
		'layers': 12,
		'max_length': 28,
		'lr': 1e-3,
		'epsilon': 0.2
	}
	
	model = RewardModel(config)
	model.train()
	optimizer = optim.Adam(model.parameters(), config['lr'])

	for epoch in range(10):
		dataset = Dataset.from_list(pairs)
		loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
		for elem in loader:
			optimizer.zero_grad()
			pos = torch.asarray(encode_batch_left_pad(elem['positive'], tokenizer, pad_token_id=tokenizer.vocab['<pad>']))
			neg = torch.asarray(encode_batch_left_pad(elem['negative'], tokenizer, pad_token_id=tokenizer.vocab['<pad>']))
			pos_logits = model(pos)
			neg_logits = model(neg)

			loss = -torch.log(torch.sigmoid(pos_logits - neg_logits)).sum()

			loss.backward()
			optimizer.step()

		print(f'epoch: {epoch}, loss: {loss.item()}')

	encoded_ids = tokenizer.encode(eval_text)
	encoded_ids = torch.asarray(encoded_ids)
	logits = model(encoded_ids.unsqueeze(0))

	print(logits.item())
