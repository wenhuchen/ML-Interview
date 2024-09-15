import numpy as np

def compute_advantages(rewards, values, gamma, lambda_):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        print('t=', t, ':', rewards[t], ' + ', gamma * next_value, ' - ', values[t], '; delta', delta)
        print('gae', lastgaelam)
        advantages[t] = lastgaelam = delta + gamma * lambda_ * lastgaelam
    return advantages

# Example usage
rewards = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
values = np.array([0.9, 0.8, 0.7, 0.3, 0.5])
gamma = 0.99
lambda_ = 0.95

advantages = compute_advantages(rewards, values, gamma, lambda_)
print(advantages)