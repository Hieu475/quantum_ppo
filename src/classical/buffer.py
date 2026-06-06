"""
buffer.py — Rollout Buffer with GAE
=====================================
Stores trajectory data (states, actions, rewards, etc.) collected during
rollouts and computes Generalized Advantage Estimation (GAE).

GAE (Schulman et al., 2016) provides a family of estimators parameterized
by λ that smoothly interpolates between high-bias (λ=0, TD residual) and
high-variance (λ=1, Monte Carlo) advantage estimates.

Formula:
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    Â_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
"""

from typing import Generator, NamedTuple

import numpy as np
import torch

from config import Config


class BatchData(NamedTuple):
    """A minibatch of training data."""
    states: torch.Tensor      # (batch, state_dim)
    actions: torch.Tensor     # (batch,)
    old_log_probs: torch.Tensor  # (batch,)
    advantages: torch.Tensor  # (batch,)
    returns: torch.Tensor     # (batch,)


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories and computing GAE advantages.

    Data is stored as lists during collection, then converted to tensors
    for batch processing during PPO updates.

    Attributes:
        config: Hyperparameter configuration.
        device: Target torch device for tensors.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = config.device
        self.clear()

    def clear(self) -> None:
        """Reset all stored data."""
        self.states: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.dones: list = []
        self.values: list = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """
        Store a single transition.

        Args:
            state: Environment observation.
            action: Selected action.
            log_prob: Log-probability of the selected action under current policy.
            reward: Reward received.
            done: Whether the episode terminated.
            value: Critic's value estimate V(s).
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value: float) -> None:
        """
        Compute Generalized Advantage Estimation and returns.

        This implements the GAE-λ estimator:
            δ_t = r_t + γ(1-d_t) V(s_{t+1}) - V(s_t)
            Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

        The returns are computed as: R_t = Â_t + V(s_t)

        Args:
            last_value: V(s_T), the value estimate of the final state.
                       Used to bootstrap the advantage computation.
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)

        # ── Backward pass through trajectory ────────────────────────────
        # Compute δ_t and accumulate GAE advantages
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t]

            # TD residual: δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)
            delta = (
                rewards[t]
                + self.config.gamma * next_value * next_non_terminal
                - values[t]
            )

            # GAE accumulation: Â_t = δ_t + γλ(1-d_t) Â_{t+1}
            last_gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * next_non_terminal
                * last_gae
            )
            advantages[t] = last_gae

        # Returns = Advantages + Values (for value function regression target)
        returns = advantages + values

        # ── Convert to tensors ───────────────────────────────────────────
        # All tensors are kept on CPU because the quantum actor (PennyLane)
        # runs on CPU. The critic handles device transfers internally.
        self._states = torch.tensor(
            np.array(self.states), dtype=torch.float32
        )
        action_dtype = (
            torch.float32
            if getattr(self.config, "action_type", "discrete") == "continuous"
            else torch.long
        )
        self._actions = torch.tensor(
            np.array(self.actions), dtype=action_dtype
        )
        self._old_log_probs = torch.tensor(
            np.array(self.log_probs), dtype=torch.float32
        )
        self._advantages = torch.tensor(
            advantages, dtype=torch.float32
        )
        self._returns = torch.tensor(
            returns, dtype=torch.float32
        )

        # ── Normalize advantages ────────────────────────────────────────
        # Normalization reduces variance and stabilizes training.
        adv_mean = self._advantages.mean()
        adv_std = self._advantages.std() + 1e-8
        self._advantages = (self._advantages - adv_mean) / adv_std

    def get_minibatches(self) -> Generator[BatchData, None, None]:
        """
        Yield shuffled minibatches for PPO updates.

        Randomly permutes the rollout data and splits into minibatches
        of size config.mini_batch_size.

        Yields:
            BatchData named tuples containing minibatch tensors.
        """
        N = len(self._states)
        indices = np.random.permutation(N)

        for start in range(0, N, self.config.mini_batch_size):
            end = start + self.config.mini_batch_size
            if end > N:
                # Skip the last incomplete minibatch
                break
            batch_idx = indices[start:end]

            yield BatchData(
                states=self._states[batch_idx],
                actions=self._actions[batch_idx],
                old_log_probs=self._old_log_probs[batch_idx],
                advantages=self._advantages[batch_idx],
                returns=self._returns[batch_idx],
            )

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.states)
