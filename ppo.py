"""
ppo.py — Proximal Policy Optimization
=======================================
Implements the PPO-Clip algorithm (Schulman et al., 2017) adapted for
the hybrid quantum-classical architecture. Uses separate optimizers for
the quantum actor and classical critic to allow independent learning
rate tuning — essential because parameter-shift gradients in quantum
circuits have different magnitude characteristics than classical autograd.

Loss components:
    L_total = -L_policy + c_v · L_value - c_e · H(π)

Where:
    L_policy = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
    L_value  = MSE(V(s), R_target)
    H(π)     = -Σ π(a|s) log π(a|s)  (entropy bonus)
"""

from typing import Dict

import torch
import torch.nn as nn

from config import Config
from agent import HybridAgent
from buffer import RolloutBuffer
from utils import compute_grad_norm


class PPO:
    """
    PPO-Clip optimizer for the hybrid quantum-classical agent.

    Manages separate Adam optimizers for the quantum actor and classical
    critic, and implements the clipped surrogate objective with entropy
    regularization.

    Attributes:
        agent: The hybrid quantum-classical agent.
        config: Hyperparameter configuration.
        actor_optimizer: Adam optimizer for quantum actor parameters.
        critic_optimizer: Adam optimizer for classical critic parameters.
    """

    def __init__(self, agent: HybridAgent, config: Config) -> None:
        self.agent = agent
        self.config = config

        # ── Separate optimizers ─────────────────────────────────────────
        # Quantum parameters need higher LR due to the bounded nature of
        # parameter-shift gradients and the typically flatter loss landscape.
        self.actor_optimizer = torch.optim.Adam(
            agent.get_actor_params(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            agent.get_critic_params(), lr=config.critic_lr
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO update over multiple epochs of minibatch updates.

        For each epoch:
        1. Shuffle and split rollout data into minibatches
        2. Compute importance sampling ratio
        3. Compute clipped policy loss
        4. Compute value loss and entropy bonus
        5. Update actor and critic with separate optimizers

        Args:
            buffer: Rollout buffer containing trajectory data with
                   pre-computed GAE advantages and returns.

        Returns:
            Dictionary of training metrics:
                - actor_loss: Mean policy loss across all updates
                - critic_loss: Mean value loss
                - entropy: Mean policy entropy
                - avg_advantage: Mean advantage value
                - actor_grad_norm: Actor gradient L2 norm
                - critic_grad_norm: Critic gradient L2 norm
                - approx_kl: Approximate KL divergence (early stopping metric)
                - clip_fraction: Fraction of ratios that were clipped
        """
        # ── Accumulators for logging ────────────────────────────────────
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_minibatches():
                # ── Evaluate current policy on stored transitions ───────
                new_log_probs, entropy, values = self.agent.evaluate_actions(
                    batch.states, batch.actions
                )

                # ── Importance sampling ratio ───────────────────────────
                # ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
                log_ratio = new_log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                # ── Approximate KL divergence ───────────────────────────
                # Used for monitoring, not for early stopping in this impl.
                # KL ≈ (ratio - 1) - log(ratio) (first-order approximation)
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # ── Clipped policy loss ─────────────────────────────────
                # L_policy = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
                advantages = batch.advantages

                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * advantages
                )
                # We take the negative because we want to MAXIMIZE the objective
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ──────────────────────────────────────────
                # MSE between predicted values and GAE returns
                value_loss = nn.functional.mse_loss(values, batch.returns)

                # ── Entropy bonus ───────────────────────────────────────
                # Encourages exploration by penalizing deterministic policies
                entropy_loss = entropy.mean()

                # ── Combined loss ───────────────────────────────────────
                # Note: policy_loss is already negated, entropy is subtracted
                # (we want to maximize entropy, so subtract it from total loss)
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy_loss
                )

                # ── Clip fraction (for monitoring) ──────────────────────
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                    .float()
                    .mean()
                    .item()
                )

                # ── Actor update ────────────────────────────────────────
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss.backward()

                # Gradient clipping for stability
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.get_actor_params(),
                    self.config.max_grad_norm,
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.get_critic_params(),
                    self.config.max_grad_norm,
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # ── Accumulate metrics ──────────────────────────────────
                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_advantage += advantages.mean().item()
                total_actor_grad_norm += (
                    actor_grad_norm.item()
                    if isinstance(actor_grad_norm, torch.Tensor)
                    else actor_grad_norm
                )
                total_critic_grad_norm += (
                    critic_grad_norm.item()
                    if isinstance(critic_grad_norm, torch.Tensor)
                    else critic_grad_norm
                )
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_updates += 1

        # ── Average metrics ─────────────────────────────────────────────
        n = max(num_updates, 1)
        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "avg_advantage": total_advantage / n,
            "actor_grad_norm": total_actor_grad_norm / n,
            "critic_grad_norm": total_critic_grad_norm / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_fraction / n,
        }
