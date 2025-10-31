"""
Problem 21: KL Divergence Calculations - Different Forms and Applications

This script demonstrates various forms of Kullback-Leibler (KL) divergence and their
applications in machine learning, particularly in post-training and RLHF contexts.

Topics covered:
1. Basic KL divergence (discrete distributions)
2. Forward KL vs Reverse KL
3. KL divergence with neural network outputs
4. Symmetric KL and Jensen-Shannon Divergence
5. Applications in RLHF/PPO (policy regularization)
6. Numerical stability considerations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class KLDivergence:
    """
    Collection of KL divergence implementations.

    KL divergence measures how one probability distribution P diverges from
    a reference distribution Q. It is defined as:

    KL(P||Q) = sum_i P(i) * log(P(i) / Q(i))

    or in continuous form:

    KL(P||Q) = integral P(x) * log(P(x) / Q(x)) dx
    """

    @staticmethod
    def forward_kl_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute forward KL divergence: KL(P||Q)

        Forward KL is "zero-forcing": prefers Q to have zero probability
        wherever P has zero probability.

        Args:
            p: Reference distribution (true distribution)
            q: Approximating distribution (model distribution)
            eps: Small constant for numerical stability

        Returns:
            KL divergence value
        """
        p = np.asarray(p)
        q = np.asarray(q)

        # Add epsilon to avoid log(0)
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)

        # Normalize to ensure valid probability distributions
        p = p / p.sum()
        q = q / q.sum()

        # KL(P||Q) = sum P(i) * log(P(i) / Q(i))
        kl = np.sum(p * np.log(p / q))

        return kl

    @staticmethod
    def reverse_kl_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute reverse KL divergence: KL(Q||P)

        Reverse KL is "zero-avoiding": prefers Q to have non-zero probability
        wherever P has non-zero probability.

        Args:
            p: Reference distribution
            q: Approximating distribution
            eps: Small constant for numerical stability

        Returns:
            Reverse KL divergence value
        """
        # Reverse KL is just swapping P and Q
        return KLDivergence.forward_kl_discrete(q, p, eps)

    @staticmethod
    def forward_kl_pytorch(p_logits: torch.Tensor, q_logits: torch.Tensor,
                          dim: int = -1) -> torch.Tensor:
        """
        Compute forward KL divergence using PyTorch: KL(P||Q)

        This version works with logits (unnormalized log probabilities).
        Commonly used in neural network training.

        Args:
            p_logits: Logits for reference distribution
            q_logits: Logits for approximating distribution
            dim: Dimension along which to compute softmax

        Returns:
            KL divergence (can be batched)
        """
        # Convert logits to log probabilities
        log_p = F.log_softmax(p_logits, dim=dim)
        log_q = F.log_softmax(q_logits, dim=dim)

        # KL(P||Q) = sum P(i) * log(P(i) / Q(i))
        #          = sum P(i) * (log P(i) - log Q(i))
        p = torch.exp(log_p)
        kl = torch.sum(p * (log_p - log_q), dim=dim)

        return kl

    @staticmethod
    def reverse_kl_pytorch(p_logits: torch.Tensor, q_logits: torch.Tensor,
                          dim: int = -1) -> torch.Tensor:
        """
        Compute reverse KL divergence using PyTorch: KL(Q||P)

        Args:
            p_logits: Logits for reference distribution
            q_logits: Logits for approximating distribution
            dim: Dimension along which to compute softmax

        Returns:
            Reverse KL divergence
        """
        return KLDivergence.forward_kl_pytorch(q_logits, p_logits, dim)

    @staticmethod
    def kl_divergence_builtin(p_logits: torch.Tensor, q_logits: torch.Tensor,
                              reduction: str = 'batchmean') -> torch.Tensor:
        """
        Compute KL divergence using PyTorch's built-in function.

        Note: PyTorch's kl_div expects log probabilities for input (Q) and
        probabilities for target (P), and computes KL(P||Q).

        Args:
            p_logits: Logits for target distribution (P)
            q_logits: Logits for input distribution (Q)
            reduction: How to reduce the result ('batchmean', 'sum', 'mean', 'none')

        Returns:
            KL divergence
        """
        log_q = F.log_softmax(q_logits, dim=-1)
        p = F.softmax(p_logits, dim=-1)

        # PyTorch's kl_div: sum P(i) * log(P(i) / Q(i))
        return F.kl_div(log_q, p, reduction=reduction)

    @staticmethod
    def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P))

        Args:
            p: First distribution
            q: Second distribution
            eps: Small constant for numerical stability

        Returns:
            Symmetric KL divergence
        """
        forward = KLDivergence.forward_kl_discrete(p, q, eps)
        reverse = KLDivergence.reverse_kl_discrete(p, q, eps)
        return 0.5 * (forward + reverse)

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """
        Compute Jensen-Shannon divergence: a symmetric divergence measure.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)

        JS divergence is bounded: 0 <= JS <= log(2) for binary distributions.

        Args:
            p: First distribution
            q: Second distribution
            eps: Small constant for numerical stability

        Returns:
            Jensen-Shannon divergence
        """
        p = np.asarray(p)
        q = np.asarray(q)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Compute mixture distribution
        m = 0.5 * (p + q)

        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        js = 0.5 * KLDivergence.forward_kl_discrete(p, m, eps) + \
             0.5 * KLDivergence.forward_kl_discrete(q, m, eps)

        return js


class RLHFKLRegularization:
    """
    Demonstrates KL divergence usage in RLHF/PPO contexts.

    In RLHF, we often regularize the policy to stay close to a reference policy
    using KL divergence penalties.
    """

    @staticmethod
    def ppo_kl_penalty(policy_logprobs: torch.Tensor,
                       ref_logprobs: torch.Tensor,
                       beta: float = 0.1) -> torch.Tensor:
        """
        Compute KL penalty for PPO-style training.

        In PPO/RLHF, we add a penalty term to prevent the policy from
        diverging too much from the reference policy:

        reward = original_reward - beta * KL(policy || ref)

        Args:
            policy_logprobs: Log probabilities from current policy
            ref_logprobs: Log probabilities from reference policy
            beta: KL penalty coefficient

        Returns:
            KL penalty term
        """
        # For sequence-level KL, we sum over tokens
        # KL = sum(policy * (log(policy) - log(ref)))
        # In log space: KL = sum(exp(log_policy) * (log_policy - log_ref))
        ratio = policy_logprobs - ref_logprobs
        kl = torch.sum(torch.exp(policy_logprobs) * ratio, dim=-1)

        return beta * kl

    @staticmethod
    def dpo_loss(policy_logprobs_chosen: torch.Tensor,
                 policy_logprobs_rejected: torch.Tensor,
                 ref_logprobs_chosen: torch.Tensor,
                 ref_logprobs_rejected: torch.Tensor,
                 beta: float = 0.1) -> torch.Tensor:
        """
        Compute DPO (Direct Preference Optimization) loss.

        DPO uses KL divergence implicitly through the log ratio formulation:

        loss = -log(σ(β * (log(π/π_ref)(y_w) - log(π/π_ref)(y_l))))

        where y_w is chosen (winner) and y_l is rejected (loser).

        Args:
            policy_logprobs_chosen: Log probs for chosen responses
            policy_logprobs_rejected: Log probs for rejected responses
            ref_logprobs_chosen: Reference log probs for chosen
            ref_logprobs_rejected: Reference log probs for rejected
            beta: Temperature parameter (inverse of KL penalty coefficient)

        Returns:
            DPO loss
        """
        # Compute log ratios
        chosen_ratio = policy_logprobs_chosen - ref_logprobs_chosen
        rejected_ratio = policy_logprobs_rejected - ref_logprobs_rejected

        # DPO objective
        logits = beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits)

        return loss.mean()

    @staticmethod
    def grpo_advantage(rewards: torch.Tensor,
                      policy_logprobs: torch.Tensor,
                      ref_logprobs: torch.Tensor,
                      beta: float = 0.1) -> torch.Tensor:
        """
        Compute advantage for GRPO (Group Relative Policy Optimization).

        GRPO uses group-relative advantages with KL regularization:
        advantage = reward - beta * KL - baseline

        Args:
            rewards: Reward values
            policy_logprobs: Log probabilities from policy
            ref_logprobs: Log probabilities from reference
            beta: KL penalty coefficient

        Returns:
            Advantages
        """
        # Compute KL penalty
        kl_penalty = policy_logprobs - ref_logprobs

        # Regularized rewards
        regularized_rewards = rewards - beta * kl_penalty.sum(dim=-1)

        # Compute advantages (normalized within group)
        advantages = regularized_rewards - regularized_rewards.mean()

        return advantages


def demonstrate_forward_vs_reverse_kl():
    """
    Demonstrate the difference between forward and reverse KL.

    Forward KL (zero-forcing): Makes Q zero where P is zero
    Reverse KL (zero-avoiding): Makes Q non-zero where P is non-zero
    """
    print("=" * 80)
    print("Forward KL vs Reverse KL")
    print("=" * 80)

    # True distribution: bimodal
    p = np.array([0.4, 0.0, 0.0, 0.0, 0.6])

    # Approximation 1: mode-seeking (puts mass on one mode)
    q1 = np.array([0.8, 0.05, 0.05, 0.05, 0.05])

    # Approximation 2: mode-covering (spreads mass)
    q2 = np.array([0.25, 0.1, 0.1, 0.1, 0.45])

    print(f"\nTrue distribution P:     {p}")
    print(f"Approximation Q1 (mode-seeking): {q1}")
    print(f"Approximation Q2 (mode-covering): {q2}")

    # Forward KL: KL(P||Q) - penalizes Q for having mass where P has none
    forward_kl_q1 = KLDivergence.forward_kl_discrete(p, q1)
    forward_kl_q2 = KLDivergence.forward_kl_discrete(p, q2)

    print(f"\nForward KL(P||Q1): {forward_kl_q1:.4f}")
    print(f"Forward KL(P||Q2): {forward_kl_q2:.4f}")
    print(f"→ Forward KL prefers Q2 (mode-covering)")

    # Reverse KL: KL(Q||P) - penalizes Q for missing mass where P has it
    reverse_kl_q1 = KLDivergence.reverse_kl_discrete(p, q1)
    reverse_kl_q2 = KLDivergence.reverse_kl_discrete(p, q2)

    print(f"\nReverse KL(Q1||P): {reverse_kl_q1:.4f}")
    print(f"Reverse KL(Q2||P): {reverse_kl_q2:.4f}")
    print(f"→ Reverse KL prefers Q1 (mode-seeking)")

    # Symmetric measures
    sym_kl_q1 = KLDivergence.symmetric_kl(p, q1)
    sym_kl_q2 = KLDivergence.symmetric_kl(p, q2)

    print(f"\nSymmetric KL with Q1: {sym_kl_q1:.4f}")
    print(f"Symmetric KL with Q2: {sym_kl_q2:.4f}")

    js_q1 = KLDivergence.js_divergence(p, q1)
    js_q2 = KLDivergence.js_divergence(p, q2)

    print(f"\nJensen-Shannon divergence with Q1: {js_q1:.4f}")
    print(f"Jensen-Shannon divergence with Q2: {js_q2:.4f}")
    print(f"→ JS divergence is symmetric and bounded")


def demonstrate_pytorch_kl():
    """
    Demonstrate KL divergence with PyTorch for neural network outputs.
    """
    print("\n" + "=" * 80)
    print("KL Divergence with PyTorch (Neural Network Outputs)")
    print("=" * 80)

    # Simulate model outputs (logits)
    batch_size = 3
    num_classes = 5

    # Reference model (e.g., teacher or previous policy)
    ref_logits = torch.randn(batch_size, num_classes)

    # Current model (e.g., student or current policy)
    policy_logits = torch.randn(batch_size, num_classes)

    print(f"\nBatch size: {batch_size}, Classes: {num_classes}")
    print(f"\nReference logits:\n{ref_logits}")
    print(f"\nPolicy logits:\n{policy_logits}")

    # Compute KL divergence
    kl_forward = KLDivergence.forward_kl_pytorch(ref_logits, policy_logits)
    kl_reverse = KLDivergence.reverse_kl_pytorch(ref_logits, policy_logits)
    kl_builtin = KLDivergence.kl_divergence_builtin(ref_logits, policy_logits,
                                                    reduction='none')

    print(f"\nForward KL(Ref||Policy) per sample: {kl_forward}")
    print(f"Reverse KL(Policy||Ref) per sample: {kl_reverse}")
    print(f"Built-in KL per sample: {kl_builtin}")

    print(f"\nMean Forward KL: {kl_forward.mean().item():.4f}")
    print(f"Mean Reverse KL: {kl_reverse.mean().item():.4f}")

    # Demonstrate gradient computation
    print("\n" + "-" * 80)
    print("Gradient Computation Example")
    print("-" * 80)

    policy_logits_grad = policy_logits.clone().requires_grad_(True)
    kl = KLDivergence.forward_kl_pytorch(ref_logits, policy_logits_grad).mean()

    print(f"\nKL divergence: {kl.item():.4f}")

    kl.backward()
    print(f"Gradient w.r.t. policy logits:\n{policy_logits_grad.grad}")
    print("→ Gradient shows direction to decrease KL (move closer to reference)")


def demonstrate_rlhf_applications():
    """
    Demonstrate KL divergence applications in RLHF.
    """
    print("\n" + "=" * 80)
    print("KL Divergence in RLHF Applications")
    print("=" * 80)

    # Simulate sequence generation scenario
    batch_size = 2
    seq_len = 5
    vocab_size = 10

    # Generate random log probabilities for demonstration
    torch.manual_seed(42)

    # For PPO/GRPO
    policy_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Convert to log probabilities
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)
    ref_logprobs = F.log_softmax(ref_logits, dim=-1)

    # Simulate rewards
    rewards = torch.tensor([1.5, -0.5])

    print("\n" + "-" * 80)
    print("PPO-style KL Penalty")
    print("-" * 80)

    beta = 0.1
    kl_penalty = RLHFKLRegularization.ppo_kl_penalty(policy_logprobs, ref_logprobs, beta)

    print(f"Rewards: {rewards}")
    print(f"KL penalty (β={beta}): {kl_penalty}")
    print(f"Regularized rewards: {rewards - kl_penalty}")
    print("→ KL penalty prevents policy from diverging too much from reference")

    print("\n" + "-" * 80)
    print("DPO Loss")
    print("-" * 80)

    # For DPO: need chosen and rejected responses
    policy_logprobs_chosen = torch.randn(batch_size, seq_len, vocab_size)
    policy_logprobs_rejected = torch.randn(batch_size, seq_len, vocab_size)
    ref_logprobs_chosen = torch.randn(batch_size, seq_len, vocab_size)
    ref_logprobs_rejected = torch.randn(batch_size, seq_len, vocab_size)

    # Sum over sequence length to get log probabilities of full sequences
    policy_chosen = F.log_softmax(policy_logprobs_chosen, dim=-1).sum(dim=1)[:, 0]
    policy_rejected = F.log_softmax(policy_logprobs_rejected, dim=-1).sum(dim=1)[:, 0]
    ref_chosen = F.log_softmax(ref_logprobs_chosen, dim=-1).sum(dim=1)[:, 0]
    ref_rejected = F.log_softmax(ref_logprobs_rejected, dim=-1).sum(dim=1)[:, 0]

    dpo_loss = RLHFKLRegularization.dpo_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        beta=0.1
    )

    print(f"DPO loss: {dpo_loss.item():.4f}")
    print("→ DPO implicitly uses KL through log ratio formulation")

    print("\n" + "-" * 80)
    print("GRPO Advantages")
    print("-" * 80)

    advantages = RLHFKLRegularization.grpo_advantage(
        rewards, policy_logprobs, ref_logprobs, beta=0.1
    )

    print(f"Rewards: {rewards}")
    print(f"Advantages (with KL regularization): {advantages}")
    print("→ Advantages are normalized and KL-regularized")


def demonstrate_numerical_stability():
    """
    Demonstrate numerical stability considerations in KL divergence.
    """
    print("\n" + "=" * 80)
    print("Numerical Stability Considerations")
    print("=" * 80)

    # Case 1: Very small probabilities
    print("\nCase 1: Very small probabilities")
    p1 = np.array([0.999, 0.001])
    q1 = np.array([0.998, 0.002])

    kl1 = KLDivergence.forward_kl_discrete(p1, q1)
    print(f"P: {p1}")
    print(f"Q: {q1}")
    print(f"KL(P||Q): {kl1:.6f}")

    # Case 2: Zero probabilities (need epsilon)
    print("\nCase 2: Near-zero probabilities (with epsilon)")
    p2 = np.array([1.0, 0.0, 0.0])
    q2 = np.array([0.99, 0.005, 0.005])

    kl2 = KLDivergence.forward_kl_discrete(p2, q2, eps=1e-10)
    print(f"P: {p2}")
    print(f"Q: {q2}")
    print(f"KL(P||Q) with eps=1e-10: {kl2:.6f}")

    # Case 3: Log-space computation (PyTorch)
    print("\nCase 3: Log-space computation (more stable)")
    p_logits = torch.tensor([[10.0, 0.0, 0.0]])
    q_logits = torch.tensor([[9.0, 0.5, 0.5]])

    kl3 = KLDivergence.forward_kl_pytorch(p_logits, q_logits)
    print(f"P logits: {p_logits.numpy()}")
    print(f"Q logits: {q_logits.numpy()}")
    print(f"KL(P||Q) in log-space: {kl3.item():.6f}")
    print("→ Working in log-space provides better numerical stability")

    # Case 4: Comparing methods
    print("\nCase 4: Consistency check between methods")
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_logits = torch.tensor([[1.5, 2.5, 2.0]])

    # Manual computation
    kl_manual = KLDivergence.forward_kl_pytorch(p_logits, q_logits)

    # Built-in PyTorch
    kl_builtin = KLDivergence.kl_divergence_builtin(p_logits, q_logits, reduction='sum')

    print(f"Manual KL: {kl_manual.item():.6f}")
    print(f"Built-in KL: {kl_builtin.item():.6f}")
    print(f"Difference: {abs(kl_manual.item() - kl_builtin.item()):.8f}")
    print("→ Both methods should give consistent results")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "KL DIVERGENCE CALCULATION DEMO" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    # Demo 1: Forward vs Reverse KL
    demonstrate_forward_vs_reverse_kl()

    # Demo 2: PyTorch KL
    demonstrate_pytorch_kl()

    # Demo 3: RLHF Applications
    demonstrate_rlhf_applications()

    # Demo 4: Numerical Stability
    demonstrate_numerical_stability()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Key Takeaways:

1. Forward KL(P||Q): Zero-forcing, mode-seeking
   - Penalizes Q for having mass where P has none
   - Used in: Maximum Likelihood Estimation, supervised learning

2. Reverse KL(Q||P): Zero-avoiding, mode-covering
   - Penalizes Q for missing mass where P has it
   - Used in: Variational Inference, RL policy optimization

3. Symmetric measures (JS divergence): Balanced
   - Treats both distributions equally
   - Used in: GANs, distribution comparison

4. In RLHF/PPO: KL regularization prevents policy drift
   - Reward = original_reward - β * KL(policy||reference)
   - Balances learning new behavior vs staying stable

5. Numerical stability: Use log-space computation
   - Avoid direct probability computation when possible
   - Add epsilon to prevent log(0)
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
