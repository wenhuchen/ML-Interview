"""
Problem 9: Reinforcement learning fine-tuning for sequence generation using a
group of rollouts, substring-match rewards with normalization, an entropy bonus
for exploration, and a policy-gradient loss; includes checkpointing and eval.
"""

import torch
from problem5 import Transformer
from torch import nn
import math
import torch.nn.functional as F

def train_data_generator(config: dict):
	batch_size: int = config['batch_size']
	vocab: int = config['vocab']
	length: int = config['length']
	inputs = torch.randint(0, vocab - 1, (batch_size, length))
	target_ = torch.max(inputs, -1).values
	target = torch.cat(
		[
			target_[:, None], 
			torch.ones_like(target_[:, None]) * (vocab - 1)
		], -1)
	return (inputs, target)


def compute_rewards(rollouts: list, target: torch.Tensor):
	advantages = []
	unnormalized_rewards = []
	for b in range(len(rollouts)):
		reward = []
		for g in range(len(rollouts[0])):
			rollout_text = str(rollouts[b][g].tolist())[1:-1]
			groundtruth = str(target[b].tolist())[1:-1]
			
			if groundtruth in rollout_text:
				reward.append(1.0)  # Full reward for exact match
			else:
				reward.append(-1.0)
		reward = torch.tensor(reward)
		reward = (reward - reward.mean()) / (reward.std() + 1e-8)
		advantages.append(reward)
		unnormalized_rewards.append(reward)

	unnormalized_rewards = torch.stack(unnormalized_rewards, 0)
	advantages = torch.stack(advantages, 0)
	return unnormalized_rewards, advantages


# Simple checkpoint functions
def save_model(model, optimizer, step):
	torch.save({
		'step': step,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict()
	}, f'checkpoint_{step}.pt')
	print(f'Saved checkpoint at step {step}')


def load_model(model, optimizer, checkpoint_path):
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	return checkpoint['step']


def compute_entropy(model, rollouts, target_length):
	# Calculate entropy bonus to encourage exploration
	logits = model(rollouts.view(-1, rollouts.shape[-1])[:, :-1])
	logits = logits[:, -target_length:, :]

	probs = F.softmax(logits, dim=-1)
	entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
	entropy = entropy.mean()  # Small entropy bonus
	return entropy


def compute_neg_logp(model, rollouts, target_length):
	batch_size, group_size = rollouts.shape[0], rollouts.shape[1]
	logits = model(rollouts.view(-1, rollouts.shape[-1]))

	# Only taking the logits of the target tokens
	logits = logits[..., :-1, :]
	logits = logits[:, -target_length:, :]
	neg_log_probs = -F.log_softmax(logits, -1)

	targets = rollouts[:, :, -target_length:]
	targets = targets.reshape(-1, targets.shape[-1])

	# Get log probabilities for the target tokens
	loss = torch.gather(neg_log_probs, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)
	
	# Reshape to [batch_size, group_size, sequence_length]
	loss = loss.view(batch_size, group_size, -1)
	return loss


def rl_step(config, model, inputs, targets):
	rollouts = []
	for i in range(config['group_size']):
		rollout = model.sample(inputs, sampling=True, temperature=1.0)
		rollouts.append(rollout)
	rollouts = torch.stack(rollouts, 1)

	# rewards: [batch, group]
	unnormalized_rewards, advantage = compute_rewards(rollouts, targets)
	
	neg_logprob = compute_neg_logp(model, rollouts, targets.shape[-1])
	entropy = compute_entropy(model, rollouts, targets.shape[-1])

	# [batch, group, length]
	adv_loss = neg_logprob * advantage.unsqueeze(-1)

	total_loss = adv_loss.sum(-1).mean(1).sum() - config['entropy_weight'] * entropy

	return unnormalized_rewards, advantage, total_loss


def evaluate(step: int, config: dict, model: nn.Module):
	inputs, targets = train_data_generator(config)
	model.eval()
	with torch.no_grad():
		y = model.sample(inputs)
	success = 0
	failure = 0
	close_matches = 0

	for i, (y_, reference_) in enumerate(zip(y, targets)):
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
	print('===============================================')
	print('Evaluation results:')
	print(f'step: {step}, success: {success/total:.3f}, close: {close_matches/total:.3f}')
	print('===============================================')


if __name__  == '__main__':
	config = {
		'hidden_dim': 64,
		'vocab': 30,
		'batch_size': 32,
		'length': 12,
		'steps': 400,
		'group_size': 50,
		'head': 8,
		'layers': 12,
		'max_length': 14,
		'entropy_weight': 1e-5,
		'lr': 1e-4,
	}

	model = Transformer(
		vocab=config['vocab'], 
		dim=config['hidden_dim'],
		head=config['head'], 
		layers=config['layers'], 
		max_length=config['max_length']
	)
	optimizer = torch.optim.Adam(model.parameters(), config['lr'])
	optimizer.zero_grad()

	for step in range(0, config['steps']):
		optimizer.zero_grad()
		inputs, targets = train_data_generator(config)
		unnormalized_rewards, _, rl_loss = rl_step(config, model, inputs, targets)
		rl_loss.backward()
		optimizer.step()
		print(f'step: {step}', rl_loss.item(), 'reward: ', unnormalized_rewards.mean().item())

		if step % 10 == 0 and step > 0:  # More frequent evaluation
			evaluate(step, config, model)

	save_model(model, optimizer, step)
