"""
classical_actor.py — Classical MLP Actor (Baseline)
=====================================================
Implements a small classical MLP policy network designed to have
approximately the same number of trainable parameters as the
Variational Quantum Circuit (VQC) actor, for a fair comparison.

The hidden layer width scales with the input dimension:
    hidden_dim = max(state_dim, 4)

This ensures correct weight matrix shapes for all environments
(CartPole: state_dim=4, LunarLander: state_dim=8, MuJoCo: state_dim>=8).

Architecture:
    State (D-dim) → Linear(D, hidden) → Tanh → Linear(hidden, action_dim) → π(a|s)

The Tanh activation is chosen to mimic the bounded output range [-1, 1]
of PauliZ measurements in the quantum circuit, ensuring structural parity.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent

from config import Config


class ClassicalActor(nn.Module):
    """
    Tiny classical MLP policy network for baseline comparison.

    This network is intentionally kept small (~30 parameters) to match
    the parameter count of the quantum actor. The goal is to provide a
    fair comparison of sample efficiency between quantum and classical
    approaches under equal parameter budgets.

    Attributes:
        network: Sequential MLP layers (Linear → Tanh → Linear).
        action_dim: Number of discrete actions.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.action_type = getattr(config, "action_type", "discrete")

        # ── Scale hidden_dim with input ───────────────────────────────────────
        # CartPole: state_dim=4 → hidden=4  (param count ≈ quantum actor)
        # LunarLander: state_dim=8 → hidden=8
        # Pendulum: state_dim=3 → hidden=4
        # Continuous envs (MuJoCo, etc.): state_dim≥17 → hidden=state_dim
        hidden_dim = max(config.state_dim, 4)

        self.feature_layer = nn.Linear(config.state_dim, hidden_dim)
        self.activation = nn.Tanh()  # Matches PauliZ output range [-1, 1]
        self.output_layer = nn.Linear(hidden_dim, config.action_dim)

        # ── Initialization ───────────────────────────────────────────────
        # Xavier for feature layer, small weights for output (stable init)
        nn.init.xavier_uniform_(self.feature_layer.weight, gain=1.0)
        nn.init.zeros_(self.feature_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

        # ── Continuous Action Space Parameters ────────────────────────────
        if self.action_type == "continuous":
            # Trainable log standard deviation for continuous actions
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

            # Buffers for action bounds scaling
            action_high = getattr(config, "action_high", None)
            action_low  = getattr(config, "action_low",  None)
            if action_high is not None and action_low is not None:
                self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
                self.register_buffer("action_low",  torch.tensor(action_low,  dtype=torch.float32))
            else:
                self.register_buffer("action_high",  torch.ones(self.action_dim, dtype=torch.float32))
                self.register_buffer("action_low",  -torch.ones(self.action_dim, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → action logits (discrete) or means (continuous).

        Args:
            state: Input state tensor, shape (state_dim,) or (batch, state_dim).

        Returns:
            Discrete:   Action logits, shape (action_dim,) or (batch, action_dim).
            Continuous: Scaled action means, shape (action_dim,) or (batch, action_dim).
        """
        x = self.feature_layer(state)
        x = self.activation(x)
        out = self.output_layer(x)

        if self.action_type == "continuous":
            # tanh squash → scale to [action_low, action_high]
            raw_mean = torch.tanh(out)
            scale  = (self.action_high - self.action_low) / 2.0
            offset = (self.action_high + self.action_low) / 2.0
            return raw_mean * scale + offset
        return out

    def get_distribution(self, state: torch.Tensor):
        """
        Get the action distribution for a given state.

        Args:
            state: Input state, shape (state_dim,) or (batch, state_dim).

        Returns:
            Categorical (discrete) or Independent(Normal) (continuous).
        """
        output = self.forward(state)
        if self.action_type == "discrete":
            return Categorical(logits=output)
        else:
            std = torch.exp(self.log_std)
            if output.dim() > 1:
                std = std.expand_as(output)
            return Independent(Normal(output, std), 1)

    def get_action_and_log_prob(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute its log-probability.

        Args:
            state: Input state, shape (state_dim,).

        Returns:
            Tuple of (action, log_probability).
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probabilities and entropy for given state-action pairs.

        Used during PPO update to compute the importance sampling ratio
        and entropy bonus.

        Args:
            states: Batch of states, shape (batch, state_dim).
            actions: Batch of actions, shape (batch,).

        Returns:
            Tuple of (log_probabilities, entropy), each shape (batch,).
        """
        dist = self.get_distribution(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
