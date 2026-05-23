"""
classical_actor.py — Classical MLP Actor (Baseline)
=====================================================
Implements a small classical MLP policy network designed to have
approximately the same number of trainable parameters as the
Variational Quantum Circuit (VQC) actor, for a fair comparison.

Parameter count comparison:
    Quantum Actor:  24 (VQC rotations) + 10 (Linear 4→2) = 34 params
    Classical Actor: 20 (Linear 4→4)   + 10 (Linear 4→2) = 30 params

Architecture:
    State (4-dim) → Linear(4, 4) → Tanh → Linear(4, 2) → Softmax → π(a|s)

The Tanh activation is chosen to mimic the bounded output range [-1, 1]
of PauliZ measurements in the quantum circuit, ensuring structural parity.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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

        # ── Tiny MLP matching quantum parameter count ───────────────────
        # Layer 1: Linear(4, 4) → 4×4 + 4 = 20 params
        # Layer 2: Linear(4, 2) → 4×2 + 2 = 10 params
        # Total: 30 params (vs. 34 in quantum actor)
        hidden_dim = 4

        self.feature_layer = nn.Linear(config.state_dim, hidden_dim)
        self.activation = nn.Tanh()  # Matches PauliZ output range [-1, 1]
        self.output_layer = nn.Linear(hidden_dim, config.action_dim)

        # ── Initialization ──────────────────────────────────────────────
        # Xavier for feature layer, small weights for output (stable init)
        nn.init.xavier_uniform_(self.feature_layer.weight, gain=1.0)
        nn.init.zeros_(self.feature_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → action logits.

        Args:
            state: Input state tensor, shape (state_dim,) or (batch, state_dim).

        Returns:
            Action logits, shape (action_dim,) or (batch, action_dim).
        """
        x = self.feature_layer(state)
        x = self.activation(x)
        logits = self.output_layer(x)
        return logits

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get the categorical action distribution for a given state.

        Args:
            state: Input state, shape (state_dim,) or (batch, state_dim).

        Returns:
            Categorical distribution over actions.
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

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
