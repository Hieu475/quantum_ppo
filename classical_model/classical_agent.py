"""
classical_agent.py — Pure Classical Agent (Baseline)
=====================================================
Mirrors the HybridAgent interface but replaces the quantum actor with
a tiny classical MLP. Uses the same critic network for fair comparison.

This agent is fully classical — no PennyLane or quantum simulation overhead,
enabling a clean apples-to-apples comparison of sample efficiency.
"""

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from config import Config
from classical_actor import ClassicalActor
from critic import Critic


class ClassicalAgent(nn.Module):
    """
    Pure classical RL agent with a tiny MLP actor and standard critic.

    Provides the same API as HybridAgent (select_action, evaluate_actions,
    get_value) so it can be used as a drop-in replacement in the PPO
    training loop.

    Attributes:
        actor: Classical MLP actor (~30 parameters).
        critic: Classical critic (same as in HybridAgent).
        device: Computation device (CPU/GPU).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.device = config.device

        # ── Sub-modules ─────────────────────────────────────────────────
        self.actor = ClassicalActor(config)
        self.critic = Critic(config)

        # Move everything to device (no PennyLane CPU constraint)
        self.actor.to(self.device)
        self.critic.to(self.device)

        # ── Log-prob clamping bounds for numerical stability ────────────
        self.log_prob_min = -20.0
        self.log_prob_max = 2.0

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Select an action during rollout collection (no gradient needed).

        Args:
            state: Raw environment observation, shape (state_dim,).

        Returns:
            Tuple of (action, log_probability, value_estimate).
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )

        # ── Actor: MLP → action distribution ────────────────────────────
        dist = self.actor.get_distribution(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ── Critic: state → value estimate ──────────────────────────────
        value = self.critic(state_tensor).squeeze(-1)

        return (
            action.item(),
            log_prob.item(),
            value.cpu().item(),
        )

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate state-action pairs for PPO update (with gradients).

        Args:
            states: Batch of states, shape (batch, state_dim).
            actions: Batch of actions, shape (batch,).

        Returns:
            Tuple of (log_probs, entropy, values).
        """
        states_device = states.to(self.device)
        actions_device = actions.to(self.device)

        # ── Actor evaluation ────────────────────────────────────────────
        log_probs, entropy = self.actor.evaluate(states_device, actions_device)

        # Clamp log-probs for numerical stability
        log_probs = torch.clamp(log_probs, self.log_prob_min, self.log_prob_max)

        # ── Critic evaluation ───────────────────────────────────────────
        values = self.critic(states_device).squeeze(-1)

        # Return on CPU for compatibility with buffer tensors
        return log_probs.cpu(), entropy.cpu(), values.cpu()

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """
        Get critic's value estimate for a single state (no gradient).

        Args:
            state: Raw environment observation, shape (state_dim,).

        Returns:
            Scalar value estimate V(s).
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )
        value = self.critic(state_tensor).squeeze(-1)
        return value.cpu().item()

    def get_actor_params(self):
        """Return actor parameters for the optimizer."""
        return self.actor.parameters()

    def get_critic_params(self):
        """Return critic parameters for the optimizer."""
        return self.critic.parameters()
