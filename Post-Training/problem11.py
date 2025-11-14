"""
Problem 11: Implement GRPO (Group Relative Policy Optimization) with an SFT
warm-up stage, clipped ratio objective against a frozen reference, and periodic
evaluation/checkpointing for a simple sequence task.
"""

# Implement GRPO Model

from problem5 import Transformer
from problem9 import train_data_generator, compute_rewards, save_model, evaluate
from problem12 import compute_neg_logp
import copy
import os
import torch
from torch import nn

if __name__  == '__main__':
    config = {
        'hidden_dim': 64,
        'vocab': 30,
        'batch_size': 32,
        'length': 12,
        'sft_steps': 50,
        'steps': 400,
        'group_size': 50,
        'head': 8,
        'layers': 12,
        'max_length': 14,
        'lr': 1e-3,
        'epsilon': 0.2
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
    loss_func = nn.CrossEntropyLoss()

    if os.path.exists(f'checkpoint_{config["sft_steps"]}.pt'):
        model.load_state_dict(torch.load('checkpoint_50.pt')['model'])
        optimizer.load_state_dict(torch.load('checkpoint_50.pt')['optimizer'])
        step = torch.load('checkpoint_50.pt')['step']
    else:
        for step in range(1, config['sft_steps'] + 1):
            optimizer.zero_grad()
            inputs, targets = train_data_generator(config)
            logits = model(torch.cat([inputs, targets], dim=1))
            logits = logits[:, inputs.shape[-1]-1:-1, :]
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), targets.view(-1))
            loss.backward()
            optimizer.step()
            print(f'step: {step}', loss.item())

            if step % 20 == 0 and step > 0:
                evaluate(step, config, model)

        print('Finished SFT Training')
    save_model(model, optimizer, step)

    optimizer = torch.optim.Adam(model.parameters(), config['lr'] * 0.5)
    optimizer.zero_grad()
    finished_steps = step
    print(f'Finished SFT Training at step {finished_steps}')

    for step in range(finished_steps, config['steps'] + finished_steps + 1):
        inputs, targets = train_data_generator(config)

        rollouts = []
        for i in range(config['group_size']):
            with torch.no_grad():
                rollout = model.sample(inputs, sampling=True, temperature=1.0)
                rollouts.append(rollout)
        rollouts = torch.stack(rollouts, 1)

        off_model = copy.deepcopy(model)
        unnormalized_rewards, advantages = compute_rewards(rollouts, targets)

        rollouts_chunks = rollouts.split(8, 0)
        advantages_chunks = advantages.split(8, 0)

        for rollouts_chunk, advantages_chunk in zip(rollouts_chunks, advantages_chunks):
            optimizer.zero_grad()
            r_theta = torch.exp(-compute_neg_logp(model, rollouts_chunk, targets.shape[-1]))
            with torch.no_grad():
                r_off = torch.exp(-compute_neg_logp(off_model, rollouts_chunk, targets.shape[-1]))
                r_off = r_off.detach()
            r = r_theta / r_off

            weighted_advantage = torch.minimum(
                r * advantages_chunk.unsqueeze(-1), 
                torch.clip(r, 1 - config['epsilon'], 1 + config['epsilon']) * advantages_chunk.unsqueeze(-1)
            )

            rl_loss = -weighted_advantage.sum(-1).mean(1).sum()
            rl_loss.backward()
            optimizer.step()

        print(f'step: {step}', rl_loss.item(), 'reward: ', unnormalized_rewards.mean().item())

        if step % 20 == 0 and step > 0:
            evaluate(step, config, model)

    save_model(model, optimizer, step)
