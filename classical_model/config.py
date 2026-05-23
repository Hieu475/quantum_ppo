"""
config.py — Centralized Hyperparameter Configuration
=====================================================
All tunable parameters are stored in a single dataclass for reproducibility
and easy hyperparameter sweeps. This is the classical-only version without
quantum-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    """Complete hyperparameter configuration for Classical PPO."""

    # ── Environment ──────────────────────────────────────────────────────
    env_name: str = "CartPole-v1"
    state_dim: int = 4          # CartPole observation space
    action_dim: int = 2         # CartPole action space (left / right)

    # ── Classical Actor ──────────────────────────────────────────────────
    actor_lr: float = 5e-3      # Learning rate for actor

    # ── Classical Critic ─────────────────────────────────────────────────
    critic_hidden: int = 64     # Hidden layer width
    critic_lr: float = 1e-3     # Standard Adam LR for classical network

    # ── PPO Algorithm ────────────────────────────────────────────────────
    gamma: float = 0.99         # Discount factor
    gae_lambda: float = 0.95    # GAE lambda for variance-bias tradeoff
    clip_epsilon: float = 0.2   # PPO surrogate clipping range
    entropy_coeff: float = 0.01 # Entropy bonus to encourage exploration
    value_coeff: float = 0.5    # Value loss weight in combined objective
    max_grad_norm: float = 0.5  # Gradient norm clipping threshold

    # ── Training ─────────────────────────────────────────────────────────
    total_timesteps: int = 100_000   # Total environment steps
    rollout_steps: int = 2048        # Steps per rollout before update
    mini_batch_size: int = 64        # Minibatch size for PPO epochs
    ppo_epochs: int = 10             # Optimization passes per rollout

    # ── Logging & Checkpoints ────────────────────────────────────────────
    log_dir: str = "runs"            # TensorBoard log directory
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 1            # Log every N episodes
    save_interval: int = 50          # Save checkpoint every N episodes
    eval_episodes: int = 10          # Episodes for periodic evaluation

    # ── Reproducibility ─────────────────────────────────────────────────
    seed: int = 42

    # ── Device ───────────────────────────────────────────────────────────
    device_str: str = "auto"         # "auto", "cpu", or "cuda"

    # ── Diagnostics ──────────────────────────────────────────────────────
    diagnose_interval: int = 10      # Run diagnostics every N updates

    @property
    def device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        assert self.rollout_steps >= self.mini_batch_size, (
            f"rollout_steps ({self.rollout_steps}) must be >= "
            f"mini_batch_size ({self.mini_batch_size})."
        )
        assert 0 < self.clip_epsilon < 1, "clip_epsilon must be in (0, 1)."
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]."
        assert 0 < self.gae_lambda <= 1, "gae_lambda must be in (0, 1]."
