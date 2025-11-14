"""
RLHF with an explicit value head (actor-critic).

Extends the Transformer with a value network for baseline estimation and
optimizes on toy preference data with PPO-style objectives.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from itertools import chain

# Reuse tokenizer and transformer
from problem2 import BPETokenizer
from problem5 import Transformer

# Reuse reward model utilities and toy preference data
from problem12 import RewardModel, encode_batch_left_pad, pairs
from torch.nn.utils.rnn import pad_sequence


class Transformer_with_value_network(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network = nn.Linear(self.dim, 1)

    def compute_value(self, x: torch.Tensor):
        last_hidden_state = self.forward(x, project=False)
        return self.value_network(last_hidden_state).squeeze(-1)


def detach_clone(model: nn.Module) -> nn.Module:
    """Create a reference copy with detached weights (no grad)."""
    ref = type(model)(
        vocab=model.vocab,
        dim=model.dim,
        head=model.layers[0].head if hasattr(model.layers[0], 'head') else 4,
        layers=len(model.layers),
        max_length=model.max_length,
        dropout=0.0,
        use_rope=model.use_rope,
    )
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def compute_logprobs(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def build_tokenizer_and_reward_model():
    tokenizer = BPETokenizer(vocab_size=100)
    corpus = chain(*[(_['positive'], _['negative']) for _ in pairs])
    tokenizer.train(corpus)
    tokenizer.expand_vocab(['<pad>'])

    config = {
        'hidden_dim': 64,
        'vocab': len(tokenizer.vocab),
        'batch_size': 32,
        'length': 12,
        'sft_steps': 50,
        'steps': 200,
        'group_size': 50,
        'head': 4,
        'layers': 4,
        'max_length': 28,
        'lr': 1e-3,
        'epsilon': 0.2,
    }

    reward_model = RewardModel(config)
    reward_model.train()
    optimizer = optim.Adam(reward_model.parameters(), config['lr'])

    # Quick reward model fitting on toy preferences
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)

    for _ in range(5):  # a few epochs is enough for demo
        # small manual batching for simplicity
        chunk = 4
        for i in range(0, len(pairs), chunk):
            batch = pairs[i:i+chunk]
            pos_texts = [b['positive'] for b in batch]
            neg_texts = [b['negative'] for b in batch]

            optimizer.zero_grad()
            pos = torch.asarray(
                encode_batch_left_pad(pos_texts, tokenizer, pad_token_id=tokenizer.vocab['<pad>'])
            )
            neg = torch.asarray(
                encode_batch_left_pad(neg_texts, tokenizer, pad_token_id=tokenizer.vocab['<pad>'])
            )
            pos_logits = reward_model(pos)
            neg_logits = reward_model(neg)
            loss = -torch.log(torch.sigmoid(pos_logits - neg_logits)).sum()
            loss.backward()
            optimizer.step()

    reward_model.eval()
    return tokenizer, config, reward_model


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lambda_: float):
    rewards_padded = torch.zeros_like(values)           # [B, T]
    rewards_padded[:, -1] = rewards                     # put reward at final step

    # Set up the advantage to zero;
    advantages = torch.zeros_like(rewards_padded)
    lastgaelam = 0
    for t in reversed(range(rewards_padded.shape[-1] - 1, -1, -1)):
        if t == rewards_padded.shape[-1] - 1:
            next_value = torch.zeros_like(values[:, t])
        else:
            next_value = values[:, t + 1]
        delta = rewards_padded[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambda_ * lastgaelam
        advantages[:, t] = lastgaelam

    return advantages


def rlhf_demo():
    tokenizer, config, reward_model = build_tokenizer_and_reward_model()
    print('Finished building tokenizer and reward model')

    # Define a tiny policy model (and a frozen reference model for KL)
    policy = Transformer_with_value_network(
        vocab=config['vocab'],
        dim=64,
        head=4,
        layers=4,
        max_length=32,
        dropout=0.0,
        use_rope=True,
    )
    policy.train()

    ref_policy = detach_clone(policy)
    ref_policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    ref_policy.to(device)
    reward_model.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Simple prompts; the goal is to answer with a positive sentiment word
    prompts = [
        "Black is",
        "White is",
        "Health is",
        "Your performance is",
        "Money is",
        "You are",
        "I am",
    ]

    pad_id = tokenizer.vocab['<pad>']
    kl_coef = 0.00
    ppo_clip = 0.2

    def encode_left_pad(texts):
        return torch.asarray(encode_batch_left_pad(texts, tokenizer, pad_token_id=pad_id))

    for step in range(500):
        # Sample a small batch of prompts
        batch_prompts = prompts[:5]
        input_ids = encode_left_pad(batch_prompts).to(device)

        # Generate a single token continuation
        with torch.no_grad():
            trajs = policy.sample(input_ids, max_new_tokens=5)

        input_actions = []
        for i, act in enumerate(trajs[:, input_ids.shape[1]:]):
            tmp = []
            for tok_id in act:
                tok_id = tok_id.item()
                word_piece = tokenizer.inv_vocab[tok_id]
                if word_piece == '<pad>':
                    break
                if '</w>' in word_piece:
                    tmp.append(tok_id)
                    break
                else:
                    tmp.append(tok_id)
            input_actions.append(
                torch.cat([trajs[i, :input_ids.shape[1]], torch.asarray(tmp)], dim=0)
            )
        input_actions = pad_sequence(input_actions, batch_first=True, padding_value=tokenizer.vocab['<pad>'])
        input_actions = input_actions.to(torch.int64)
        actions = input_actions[:, input_ids.shape[1]:]

        # Compute policy and ref logprobs for chosen actions
        logprob_pi = policy(input_actions)
        logprob_pi = logprob_pi[:, input_ids.shape[1] - 1: - 1, :]
        logprob_pi_chosen = logprob_pi.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            logprob_ref = ref_policy(input_actions)
            logprob_ref = logprob_ref[:, input_ids.shape[1] - 1: - 1, :]
            logprob_ref_chosen = logprob_ref.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Build full texts to score with reward model
        decoded = []
        for i, prompt in enumerate(batch_prompts):
            token = actions[i].tolist()
            text = prompt + " " + tokenizer.decode(token)
            decoded.append(text)

        with torch.no_grad():
            rm_inputs = torch.asarray(encode_batch_left_pad(decoded, tokenizer, pad_token_id=pad_id)).to(device)
            rewards = reward_model(rm_inputs)

        # Estimate the value of each step
        values = policy.compute_value(input_actions)
        values = values[:, input_ids.shape[1] - 1:-1]
        value_loss = F.mse_loss(values, rewards.unsqueeze(-1))

        advantages = compute_advantages(rewards, values, 0.99, 0.95)

        # KL penalty using action-level divergence
        kl = (torch.exp(logprob_pi) * (logprob_pi - logprob_ref.detach())).sum(-1)
        advantages_with_kl = advantages - kl_coef * kl

        # PPO surrogate loss (single step)
        ratio = torch.exp(logprob_pi_chosen - logprob_ref_chosen.detach())

        unclipped = ratio * advantages_with_kl
        clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_with_kl
        ppo_loss = -torch.mean(torch.min(unclipped, clipped)) + torch.mean(value_loss)

        optimizer.zero_grad()
        ppo_loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            avg_reward = rewards.mean().item()
            print(decoded)
            print(f"Step {step+1}: ppo_loss={ppo_loss.item():.4f}, value_loss={value_loss.item():.4f}, reward={avg_reward:.4f}")

    # Show qualitative generations after RLHF
    policy.eval()
    tests = ["Black is", "White is", "Health is", "You are", "I am"]
    with torch.no_grad():
        x = encode_left_pad(tests).to(device)
        trajs = policy.sample(x, max_new_tokens=5)
        for i, traj in enumerate(trajs):
            print(tokenizer.decode(traj.tolist()))


if __name__ == "__main__":
    rlhf_demo()
