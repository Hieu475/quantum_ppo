"""
critic.py — Classical Critic Network
======================================
A standard MLP that estimates the state-value function V(s).
Uses orthogonal initialization for training stability, which is
a well-established practice in deep RL (Andrychowicz et al., 2020).

Architecture:
    State (4-dim) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1) → V(s)
"""

import numpy as np
import torch
import torch.nn as nn

from config import Config


class Critic(nn.Module):
    """
    Classical critic network for state-value estimation.

    A 2-layer MLP that maps states to scalar value estimates V(s).
    The critic is trained purely classically with standard backpropagation,
    while the quantum actor uses parameter-shift gradients.

    Attributes:
        network: Sequential MLP layers.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.critic_hidden),
            nn.ReLU(),
            nn.Linear(config.critic_hidden, config.critic_hidden),
            nn.ReLU(),
            nn.Linear(config.critic_hidden, 1),
        )

        # ── Orthogonal initialization ───────────────────────────────────
        # Preserves gradient magnitude across layers, preventing
        # vanishing/exploding gradients in deep networks.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization to linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Last layer gets smaller gain for stable initial value predictions
        last_linear = [m for m in self.network if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V(s).

        Args:
            state: State tensor, shape (state_dim,) or (batch, state_dim).

        Returns:
            Value estimate, shape (1,) or (batch, 1).
        """
        return self.network(state)


