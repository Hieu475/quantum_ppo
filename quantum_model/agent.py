"""
agent.py — Hybrid Quantum-Classical Agent
==========================================
Unifies the Quantum Actor and Classical Critic into a single interface
for rollout collection and PPO evaluation. Handles device transfers,
numerical stability (log-prob clamping), and provides clean APIs for
the training loop.

Now accepts obs_space to support generalized state encoding, allowing
the actor and critic to auto-adapt to any Gymnasium observation space.

The agent does NOT own the optimizers — those live in the PPO module
to maintain separation of concerns.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import numpy as np

import gymnasium as gym

from config import Config
from quantum_actor import QuantumActor
from critic import Critic


class HybridAgent(nn.Module):
    """
    Hybrid RL agent combining a quantum actor and classical critic.

    The actor produces a policy π(a|s) via a variational quantum circuit,
    while the critic estimates V(s) via a classical MLP (or CNN+MLP for
    image observations). Both are wrapped in a single nn.Module for
    convenient parameter access.

    Attributes:
        actor: Quantum actor (VQC-based policy) with pre-encoding NN.
        critic: Classical critic (MLP/CNN-based value function).
        device: Computation device (CPU/GPU).
    """

    def __init__(self, config: Config, obs_space: gym.spaces.Space) -> None:
        super().__init__()
        self.config = config
        self.device = config.device

        # ── Sub-modules ─────────────────────────────────────────────────
        self.actor = QuantumActor(config, obs_space)
        self.critic = Critic(config, obs_space)

        # Move critic to device (actor runs on CPU via PennyLane simulator)
        self.critic.to(self.device)

        # ── Log-prob clamping bounds for numerical stability ────────────
        # Prevents log(0) = -inf which causes NaN in ratio computation
        self.log_prob_min = -20.0
        self.log_prob_max = 2.0

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray
    ) -> Tuple[Union[int, np.ndarray], float, float]:
        """
        Select an action during rollout collection (no gradient needed).

        Args:
            state: Raw environment observation, shape (state_dim,).

        Returns:
            Tuple of (action, log_probability, value_estimate).
            - action: Integer action or Numpy array for the environment.
            - log_probability: log π(a|s), used as old_log_prob in PPO.
            - value_estimate: V(s) from the critic.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # ── Actor: quantum circuit → action distribution ────────────────
        dist = self.actor.get_distribution(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ── Critic: state → value estimate ──────────────────────────────
        state_device = state_tensor.to(self.device)
        value = self.critic(state_device).squeeze(-1)

        if getattr(self.config, "action_type", "discrete") == "discrete":
            action_out = action.item()
        else:
            action_out = action.cpu().numpy()

        return (
            action_out,
            log_prob.item(),
            value.cpu().item(),
        )

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate state-action pairs for PPO update (with gradients).

        This method is called during the PPO optimization loop and must
        maintain gradient flow through both actor and critic.

        Args:
            states: Batch of states, shape (batch, state_dim).
            actions: Batch of actions, shape (batch,).

        Returns:
            Tuple of (log_probs, entropy, values):
            - log_probs: log π_new(a|s), shape (batch,).
            - entropy: H(π(·|s)), shape (batch,).
            - values: V(s), shape (batch,).
        """
        # ── Actor evaluation ────────────────────────────────────────────
        # PennyLane quantum circuits run on CPU, so we must ensure
        # states and actions are on CPU for actor evaluation.
        states_cpu = states.cpu()
        actions_cpu = actions.cpu()
        log_probs, entropy = self.actor.evaluate(states_cpu, actions_cpu)

        # Clamp log-probs for numerical stability
        log_probs = torch.clamp(log_probs, self.log_prob_min, self.log_prob_max)

        # ── Critic evaluation ───────────────────────────────────────────
        states_device = states.to(self.device)
        values = self.critic(states_device).squeeze(-1)

        # Ensure all outputs are on the same device (CPU for compatibility)
        return log_probs, entropy, values.cpu()

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """
        Get critic's value estimate for a single state (no gradient).

        Used at the end of a rollout to bootstrap GAE computation.

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
