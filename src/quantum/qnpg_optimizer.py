"""
qnpg_optimizer.py — Quantum Natural Policy Gradient (QNPG) Optimizer
=====================================================================
Implements QNPG as a custom PyTorch optimizer that wraps standard
gradient computation with QFIM-preconditioning for quantum actor parameters.

The key insight is that VQC parameters live on a **Riemannian manifold**
(the space of quantum states, equipped with the Fubini-Study metric), not
a flat Euclidean space. Standard gradient descent ignores this geometry,
leading to:
    - Inefficient parameter updates (stepping in wrong directions)
    - Poor sample efficiency (needing more environment interactions)
    - Sensitivity to parameterization choices

QNPG corrects for this by using the **quantum Fisher information matrix**
as a metric tensor, yielding updates that are covariant under re-parameterizations.

Algorithm
---------
For each PPO update step:
    1. Compute Euclidean gradient g = ∇_θ L(θ) via standard backprop
    2. Estimate QFIM F(θ) using sampled states from the replay buffer
    3. Compute natural gradient: g_nat = F(θ)^{-1} · g
    4. Optionally clip natural gradient for stability
    5. Update: θ ← θ - η · g_nat

Integration with PPO
--------------------
QNPG only modifies the **actor** (quantum) parameter update. The critic
continues using standard Adam, as it has no quantum parameters.

References
----------
[1] Stokes et al. (2020). "Quantum Natural Gradient." Quantum 4, 269.
[2] Meyer et al. (2023). "Quantum Natural Policy Gradients: Towards
    Sample-Efficient Reinforcement Learning." QIP 2023.
[3] Schulman et al. (2015). "Trust Region Policy Optimization." ICML 2015.
    (Classical NPG motivation)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from qfim import QFIMEstimator, QFIMMode
from quantum_actor import QuantumActor


class QNPGOptimizer:
    """
    Quantum Natural Policy Gradient optimizer for VQC actor parameters.

    This optimizer wraps the QFIM estimator and applies natural gradient
    preconditioning to the quantum actor's parameter updates. It is designed
    to be used **alongside** a standard Adam optimizer for classical parameters.

    Key behaviors:
    - Only quantum parameters (q_params, input_scaling) are preconditioned
    - Classical compression and post-processing layers use standard Adam
    - QFIM estimation is amortized: computed once per PPO rollout update
    - Gradient clipping is applied after preconditioning for stability

    Attributes:
        actor: The VQC-based quantum actor.
        lr: Learning rate for natural gradient updates.
        qfim_mode: QFIM approximation mode (diagonal/block_diagonal/full).
        damping: QFIM regularization constant δ for numerical stability.
        n_qfim_samples: Number of states used for QFIM Monte Carlo estimation.
        max_grad_norm: Clipping threshold for natural gradient norm.
        use_natural_grad: Toggle to disable QNPG (fall back to SGD) for ablation.
    """

    def __init__(
        self,
        actor: QuantumActor,
        lr: float = 5e-3,
        qfim_mode: QFIMMode = QFIMMode.DIAGONAL,
        damping: float = 1e-3,
        n_qfim_samples: int = 4,
        max_grad_norm: float = 0.5,
        use_natural_grad: bool = True,
        block_size: int = 9,
        max_precondition_ratio: float = 10.0,
        warmup_steps: int = 0,
    ) -> None:
        self.actor = actor
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.use_natural_grad = use_natural_grad
        # Safety clamp: nat_grad_norm / euc_grad_norm <= max_precondition_ratio
        # Prevents QFIM≈0 → F⁻¹g = g/δ blowing up when damping is small.
        self.max_precondition_ratio = max_precondition_ratio
        # Warm-up: use plain Euclidean gradient for the first N steps
        # before switching to QNPG. Allows QFIM to become non-trivial.
        self.warmup_steps = warmup_steps
        self._step_count = 0

        # QFIM Estimator
        self.qfim_estimator = QFIMEstimator(
            actor=actor,
            mode=qfim_mode,
            damping=damping,
            block_size=block_size,
            n_samples=n_qfim_samples,
        )

        # Metrics storage for logging
        self._last_nat_grad_norm: float = 0.0
        self._last_euc_grad_norm: float = 0.0
        self._last_qfim_stats: dict = {}

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════════

    def zero_grad(self) -> None:
        """Zero gradients of quantum actor parameters."""
        for p in self._get_quantum_params():
            if p.grad is not None:
                p.grad.zero_()

    def step(self, states: torch.Tensor) -> dict:
        """
        Apply QNPG update to quantum actor parameters.

        This method:
        1. Extracts current Euclidean gradients from quantum params
        2. Computes/estimates QFIM using provided state samples
        3. Computes natural gradient via QFIM preconditioning
        4. Clips natural gradient norm
        5. Applies in-place parameter update

        Must be called AFTER loss.backward() so that gradients are populated.

        Args:
            states: Batch of environment states for QFIM estimation.
                    Shape: (batch_size, obs_dim).

        Returns:
            Dictionary of optimizer metrics:
                - euclidean_grad_norm: L2 norm of raw gradient
                - natural_grad_norm: L2 norm after preconditioning
                - qfim_mean_diag: Mean QFIM diagonal (avg. sensitivity)
                - qfim_condition: Approximate condition number of QFIM
                - qfim_effective_dim: Number of "active" quantum parameters
        """
        # ── Step 1: Collect Euclidean gradients ──────────────────────────
        quantum_params = list(self._get_quantum_params())
        grads = []

        for p in quantum_params:
            if p.grad is not None:
                grads.append(p.grad.flatten().clone())
            else:
                grads.append(torch.zeros(p.numel(), dtype=torch.float32))

        if not grads:
            return {}

        euclidean_grad = torch.cat(grads)  # (d,)
        euc_norm = euclidean_grad.norm().item()
        self._last_euc_grad_norm = euc_norm

        # ── Step 2: Compute QFIM stats (for logging only) ─────────────────
        # Only run full stats every Nth call to reduce overhead
        with torch.no_grad():
            n = min(self.qfim_estimator.n_samples, states.shape[0])
            idx = torch.randperm(states.shape[0])[:n]
            sample_states = states[idx].float()
            qfim_stats = self.qfim_estimator.get_qfim_stats(sample_states)
        self._last_qfim_stats = qfim_stats

        # ── Step 3: Compute natural gradient ──────────────────────────────
        self._step_count += 1
        in_warmup = self._step_count <= self.warmup_steps

        if self.use_natural_grad and not in_warmup:
            nat_grad = self.qfim_estimator.compute_natural_gradient(
                sample_states, euclidean_grad
            )
            # ── Safety clamp: prevent QFIM≈0 → huge amplification ─────────
            # When QFIM diagonal ≈ 0 (untrained network), F⁻¹g = g/δ can
            # amplify by 1/damping (e.g. ×1000). We cap the ratio at
            # max_precondition_ratio to bound the effective step size.
            if euc_norm > 1e-10:
                nat_norm_raw = nat_grad.norm().item()
                ratio = nat_norm_raw / euc_norm
                if ratio > self.max_precondition_ratio:
                    nat_grad = nat_grad * (self.max_precondition_ratio / ratio)
        else:
            # Warm-up or ablation: use plain Euclidean gradient
            nat_grad = euclidean_grad

        # ── Step 4: Clip natural gradient ─────────────────────────────────
        nat_norm = nat_grad.norm()
        if nat_norm > self.max_grad_norm:
            nat_grad = nat_grad * (self.max_grad_norm / (nat_norm + 1e-8))
        self._last_nat_grad_norm = nat_grad.norm().item()

        # ── Step 5: Apply updates in-place ────────────────────────────────
        offset = 0
        for p in quantum_params:
            n_params = p.numel()
            p_grad = nat_grad[offset: offset + n_params].reshape(p.shape)
            p.data -= self.lr * p_grad
            offset += n_params

        return {
            "euclidean_grad_norm": euc_norm,
            "natural_grad_norm": self._last_nat_grad_norm,
            "qfim_mean_diag": qfim_stats.get("mean_diag", 0.0),
            "qfim_condition": qfim_stats.get("condition_number", 0.0),
            "qfim_effective_dim": qfim_stats.get("effective_dim", 0),
        }

    def update_lr(self, new_lr: float) -> None:
        """Update the learning rate (for LR scheduling)."""
        self.lr = new_lr

    def get_last_metrics(self) -> dict:
        """Return cached metrics from the last step() call."""
        return {
            "euclidean_grad_norm": self._last_euc_grad_norm,
            "natural_grad_norm": self._last_nat_grad_norm,
            **self._last_qfim_stats,
        }

    # ════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _get_quantum_params(self):
        """
        Yield only the quantum parameters (q_params and input_scaling).
        
        We deliberately exclude classical_compression and post_nn parameters,
        as those are updated by the standard Adam optimizer in PPO.
        """
        yield self.actor.q_params
        if isinstance(self.actor.input_scaling, nn.Parameter):
            yield self.actor.input_scaling
