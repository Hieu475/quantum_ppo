"""
utils.py — Utility Functions
=============================
Provides reproducibility helpers, gradient diagnostics, and monitoring
tools specifically designed for hybrid quantum-classical training where
barren plateaus and vanishing gradients are primary concerns.
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

    Used for monitoring training stability. Exploding gradients (>10)
    or vanishing gradients (<1e-7) both indicate problems.

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


def diagnose_barren_plateau(model: nn.Module, threshold: float = 1e-6) -> dict:
    """
    Analyze quantum parameter gradients for barren plateau symptoms.

    Barren plateaus manifest as exponentially vanishing gradients in
    parameterized quantum circuits. This function checks gradient
    magnitudes and provides diagnostic information.

    Args:
        model: The quantum actor module (nn.Module).
        threshold: Gradient magnitude below which parameters are
                   considered "barren" (stuck in flat landscape).

    Returns:
        Dictionary with diagnostic results:
            - total_params: Number of quantum parameters
            - vanishing_count: Params with |grad| < threshold
            - mean_grad_magnitude: Average |grad| across params
            - max_grad_magnitude: Maximum |grad| across params
            - is_barren: Whether barren plateau is detected
    """
    grad_magnitudes = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            mag = p.grad.data.abs().mean().item()
            grad_magnitudes.append(mag)

    if not grad_magnitudes:
        return {
            "total_params": 0,
            "vanishing_count": 0,
            "mean_grad_magnitude": 0.0,
            "max_grad_magnitude": 0.0,
            "is_barren": True,
            "message": "No gradients found — parameters may not be connected to loss.",
        }

    mean_mag = np.mean(grad_magnitudes)
    max_mag = np.max(grad_magnitudes)
    vanishing = sum(1 for m in grad_magnitudes if m < threshold)
    total = len(grad_magnitudes)

    is_barren = vanishing > total * 0.5  # >50% params have vanishing grads

    message = "OK"
    if is_barren:
        message = (
            f"⚠️  Barren plateau detected: {vanishing}/{total} parameters have "
            f"|grad| < {threshold:.1e}. Consider: (1) reducing circuit depth, "
            f"(2) using local cost functions, (3) layer-wise training."
        )
    elif mean_mag < threshold * 10:
        message = (
            f"⚡ Gradients are very small (mean={mean_mag:.2e}). "
            f"Training may be slow."
        )

    return {
        "total_params": total,
        "vanishing_count": vanishing,
        "mean_grad_magnitude": mean_mag,
        "max_grad_magnitude": max_mag,
        "is_barren": is_barren,
        "message": message,
    }


def diagnose_entropy(entropy: float, action_dim: int = 2, action_type: str = "discrete") -> str:
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
    if action_type == "continuous":
        return f"ℹ️  Continuous entropy ({entropy:.4f}). Monitoring std via TensorBoard."
        
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
