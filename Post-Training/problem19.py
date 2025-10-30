"""
Direct Preference Optimization (DPO) demo.

Trains a policy against a frozen reference on toy preference pairs using a
beta-weighted logistic objective, then samples from the trained policy.
"""

from problem2 import BPETokenizer
from itertools import chain
from problem5 import Transformer
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from torch import optim
from problem13 import detach_clone

pairs = [
	{"positive": f"[BOS] You are good", "negative": f"[BOS] You are bad"},
	{"positive": f"[BOS] I am good", "negative": f"[BOS] I am bad"},
	{"positive": f"[BOS] Your performance is good", "negative": f"[BOS] Your performance is bad"},
	{"positive": f"[BOS] Your money is good", "negative": f"[BOS] Your money is bad"},
	{"positive": f"[BOS] Your health is good", "negative": f"[BOS] Your health is bad"},
	{"positive": f"[BOS] Health is good", "negative": f"[BOS] Your health is bad"},
	{"positive": f"[BOS] White is good", "negative": f"[BOS] Black is bad"},
	{"positive": f"[BOS] White is good", "negative": f"[BOS] White is bad"},
	{"positive": f"[BOS] Black is good", "negative": f"[BOS] White is bad"},
	{"positive": f"[BOS] Black is good", "negative": f"[BOS] Black is bad"}
]


eval_text = f"[BOS] "
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

if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=100)
    corpus = chain(*[(_['positive'], _['negative']) for _ in pairs])
    tokenizer.train(corpus)
    tokenizer.expand_vocab(['<pad>'])
    print(tokenizer.vocab)

    config = {
        'hidden_dim': 64,
        'vocab': len(tokenizer.vocab),
        'batch_size': 4,
        'length': 12,
        'steps': 400,
        'group_size': 50,
        'head': 8,
        'layers': 12,
        'max_length': 28,
        'lr': 1e-3,
        'epsilon': 0.2,
        'beta': 0.2,
    }

    policy = Transformer(
        vocab=config['vocab'],
        dim=config['hidden_dim'],
        head=config['head'],
        layers=config['layers'],
        max_length=config['max_length'],
        dropout=0.1,
        use_rope=True
    )
    ref_policy = detach_clone(policy)
    ref_policy.eval()

    optimizer = optim.Adam(policy.parameters(), config['lr'])
    optimizer.zero_grad()

    dataset = Dataset.from_list(pairs)

    for epoch in range(40):
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        for batch in loader:
            pos = torch.asarray(
                encode_batch_left_pad(batch['positive'], tokenizer, pad_token_id=tokenizer.vocab['<pad>']))
            neg = torch.asarray(
                encode_batch_left_pad(batch['negative'], tokenizer, pad_token_id=tokenizer.vocab['<pad>']))
            pos_logprobs = torch.log_softmax(policy(pos), dim=-1)
            neg_logprobs = torch.log_softmax(policy(neg), dim=-1)
            pos_logprobs_chosen = pos_logprobs[:, :-1, :].gather(-1, pos[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
            neg_logprobs_chosen = neg_logprobs[:, :-1, :].gather(-1, neg[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)

            with torch.no_grad():
                ref_pos_logprobs = torch.log_softmax(ref_policy(pos), dim=-1)
                ref_neg_logprobs = torch.log_softmax(ref_policy(neg), dim=-1)
                ref_pos_logprobs_chosen = ref_pos_logprobs[:, :-1, :].gather(-1, pos[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
                ref_neg_logprobs_chosen = ref_neg_logprobs[:, :-1, :].gather(-1, neg[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)

            loss = -torch.log(
                torch.sigmoid(config['beta'] * (
                        pos_logprobs_chosen - neg_logprobs_chosen - ref_pos_logprobs_chosen + ref_neg_logprobs_chosen
                    )
                )
            )

            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch: {epoch}, loss: {loss.item()}')

    encoded_ids = tokenizer.encode(eval_text)
    encoded_ids = torch.asarray(encoded_ids)
    trajs = policy.sample(encoded_ids.unsqueeze(0))
    trajs = trajs[0].tolist()
    print(tokenizer.decode(trajs))
