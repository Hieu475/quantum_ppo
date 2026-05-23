"""
utils.py — Utility Functions (Classical)
==========================================
Provides reproducibility helpers, gradient diagnostics, and monitoring
tools for classical PPO training.
"""

import random
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.

    Args:
        device_str: One of "auto", "cpu", "cuda".

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute the global L2 gradient norm across all parameters.

    Args:
        model: PyTorch module.

    Returns:
        Global L2 norm of all gradients, or 0.0 if no gradients exist.
    """
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
            param_count += 1
    if param_count == 0:
        return 0.0
    return total_norm ** 0.5


def diagnose_entropy(entropy: float, action_dim: int = 2) -> str:
    """
    Diagnose policy entropy health.

    For CartPole (2 actions), maximum entropy = ln(2) ≈ 0.693.
    - Entropy near max → policy is nearly uniform (too random).
    - Entropy near 0   → policy has collapsed (no exploration).
    - Healthy range: 0.1 to 0.6 for CartPole.

    Args:
        entropy: Current policy entropy value.
        action_dim: Number of discrete actions.

    Returns:
        Diagnostic message string.
    """
    max_entropy = np.log(action_dim)
    ratio = entropy / max_entropy if max_entropy > 0 else 0.0

    if ratio > 0.95:
        return (
            f"⚠️  Entropy too high ({entropy:.4f}/{max_entropy:.4f}). "
            f"Policy is nearly uniform — agent is not learning."
        )
    elif ratio < 0.05:
        return (
            f"⚠️  Entropy collapsed ({entropy:.4f}/{max_entropy:.4f}). "
            f"Policy is deterministic — no exploration. "
            f"Increase entropy_coeff or check for NaN."
        )
    elif ratio < 0.15:
        return (
            f"⚡ Low entropy ({entropy:.4f}/{max_entropy:.4f}). "
            f"Exploration is limited."
        )
    else:
        return f"✅ Entropy OK ({entropy:.4f}/{max_entropy:.4f}, ratio={ratio:.2f})."


def make_env(env_name: str, seed: int) -> gym.Env:
    """
    Create and seed a Gymnasium environment.

    Args:
        env_name: Environment ID string.
        seed: Random seed.

    Returns:
        Seeded Gymnasium environment.
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env
