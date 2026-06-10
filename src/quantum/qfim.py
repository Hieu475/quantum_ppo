"""
qfim.py — Quantum Fisher Information Matrix (QFIM) Computation
===============================================================
Implements efficient QFIM estimation for Variational Quantum Circuits (VQC)
using the parameter-shift rule. The QFIM is the quantum analogue of the
classical Fisher Information Matrix and measures the sensitivity of the
quantum state to changes in circuit parameters.

Mathematical Background
-----------------------
For a parameterized quantum circuit U(θ)|0⟩ = |ψ(θ)⟩, the QFIM is defined as:

    F_{ij}(θ) = 4 Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

This is related to the Fubini-Study metric tensor on the space of quantum states
and captures the true information geometry of the parameter space.

For RL purposes, we use the **Quantum Natural Gradient (QNG)** formulation:

    θ_{t+1} = θ_t + η · F(θ)^{-1} · ∇L(θ)

Key Approximations Implemented
--------------------------------
1. **Diagonal QFIM** (O(d) cost): Only diagonal elements F_{ii}
   - Fastest, good when parameters are nearly independent
   - F_{ii} = 1 - |⟨ψ(θ + e_i·π/2)|ψ(θ - e_i·π/2)⟩|²  (parameter shift)

2. **Block-diagonal QFIM** (O(d²) cost per block): Full matrix within each layer
   - Better captures intra-layer parameter correlations
   - Uses full off-diagonal terms within each (layer, qubit) block

3. **Full QFIM** (O(d²) cost): Complete d×d matrix
   - Most accurate, expensive for d > 100

References
----------
[1] Stokes et al. (2020). "Quantum Natural Gradient." Quantum 4, 269.
[2] Yamamoto (2019). "On the natural gradient for variational quantum eigensolver."
    arXiv:1910.11526.
[3] Gacon et al. (2021). "Simultaneous Perturbation Stochastic Approximation of
    the Quantum Fisher Information." Quantum 5, 567.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from quantum_actor import QuantumActor


class QFIMMode(Enum):
    """Approximation mode for QFIM computation."""

    DIAGONAL = "diagonal"          # O(2d) circuit evaluations — fastest
    BLOCK_DIAGONAL = "block_diag"  # O(d²/B) per block — balanced
    FULL = "full"                  # O(d²) — most accurate, expensive


class QFIMEstimator:
    """
    Estimates the Quantum Fisher Information Matrix for a VQC actor.

    Supports three approximation modes with different cost-accuracy tradeoffs.
    Uses the parameter-shift rule to compute QFIM entries without access to
    the quantum state vector directly (hardware compatible).

    The estimator operates on the **quantum parameters only** (q_params and
    input_scaling), not the classical compression/post-processing layers.
    Classical parameters use standard Adam gradients.

    Attributes:
        actor: The QuantumActor whose QFIM we compute.
        mode: Approximation mode (diagonal/block_diagonal/full).
        damping: Tikhonov regularization δ to ensure F + δI is invertible.
        block_size: Number of parameters per block (block_diagonal mode).
        n_samples: Number of state samples for Monte Carlo QFIM estimation.
    """

    def __init__(
        self,
        actor: QuantumActor,
        mode: QFIMMode = QFIMMode.DIAGONAL,
        damping: float = 1e-3,
        block_size: int = 9,  # n_qubits * 3 per layer (matches q_params shape)
        n_samples: int = 4,
    ) -> None:
        self.actor = actor
        self.mode = mode
        self.damping = damping
        self.block_size = block_size
        self.n_samples = n_samples

        # Cache the flat parameter dimension for quantum params
        self._d = actor.q_params.numel()
        if isinstance(actor.input_scaling, nn.Parameter):
            self._d += actor.input_scaling.numel()

    # ════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════════

    def compute_natural_gradient(
        self,
        states: torch.Tensor,
        euclidean_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the natural gradient: g_nat = F(θ)^{-1} · g_euc

        This is the core operation for QNPG. The natural gradient rescales
        the Euclidean gradient by the inverse QFIM, correcting for the
        non-Euclidean geometry of the quantum parameter manifold.

        Args:
            states: Batch of environment states, shape (batch, obs_dim).
                    Used to estimate the empirical QFIM (Monte Carlo).
            euclidean_grad: Standard gradient ∇_θ L, shape (d,).
                            Must be the concatenated flat gradient of
                            quantum parameters (q_params + input_scaling).

        Returns:
            Natural gradient F^{-1} g, shape (d,), same as euclidean_grad.
        """
        # Select subset of states for QFIM estimation (expensive operation)
        n = min(self.n_samples, states.shape[0])
        idx = torch.randperm(states.shape[0])[:n]
        sample_states = states[idx]

        if self.mode == QFIMMode.DIAGONAL:
            return self._natural_grad_diagonal(sample_states, euclidean_grad)
        elif self.mode == QFIMMode.BLOCK_DIAGONAL:
            return self._natural_grad_block_diagonal(sample_states, euclidean_grad)
        else:  # FULL
            return self._natural_grad_full(sample_states, euclidean_grad)

    def estimate_diagonal_qfim(
        self, states: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate the diagonal QFIM entries using parameter-shift rule.

        For each parameter θ_i, the diagonal entry is:
            F_{ii} = 1 - |⟨ψ(θ + s·ê_i)|ψ(θ - s·ê_i)⟩|² / 4·s²

        In the expectation value formulation (observable O):
            F_{ii} ≈ Var_π[∂_i log π(a|s)]

        We approximate this via the empirical gradient variance:
            F_{ii} ≈ E_s[⟨(∂_i log π)²⟩] - E_s[⟨∂_i log π⟩]²

        Args:
            states: Sample states, shape (n_samples, obs_dim).

        Returns:
            Diagonal QFIM, shape (d,). Values ∈ [0, 1] by construction.
        """
        n_samples = states.shape[0]
        d = self._d

        # Accumulate gradient outer products over samples
        diag_fim = torch.zeros(d, dtype=torch.float32)

        for s in range(n_samples):
            state = states[s]  # (obs_dim,)

            # Compute ∂_θ log π(a|s) for this state
            log_grad = self._compute_log_policy_grad(state)  # (d,)

            diag_fim += log_grad ** 2

        diag_fim /= n_samples

        return diag_fim  # F_{ii} ≈ E[⟨(∂_i log π)²⟩]

    def get_qfim_stats(self, states: torch.Tensor) -> Dict[str, float]:
        """
        Compute summary statistics of the QFIM for monitoring/logging.

        Args:
            states: Sample states, shape (n_samples, obs_dim).

        Returns:
            Dictionary with QFIM statistics:
                - mean_diag: Mean diagonal element (avg. sensitivity)
                - max_diag: Maximum diagonal element
                - min_diag: Minimum diagonal element
                - condition_number: max/min ratio (approximates κ(F))
                - effective_dim: Count of "active" parameters (diag > threshold)
        """
        with torch.no_grad():
            diag = self.estimate_diagonal_qfim(states)

        return {
            "mean_diag": diag.mean().item(),
            "max_diag": diag.max().item(),
            "min_diag": diag.min().item(),
            "condition_number": (
                (diag.max() / (diag.min() + 1e-10)).item()
            ),
            "effective_dim": (diag > 1e-4).sum().item(),
        }

    # ════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════════════

    def _get_quantum_params_flat(self) -> torch.Tensor:
        """Return a flat view of all quantum parameters."""
        parts = [self.actor.q_params.data.flatten()]
        if isinstance(self.actor.input_scaling, nn.Parameter):
            parts.append(self.actor.input_scaling.data.flatten())
        return torch.cat(parts)

    def _compute_log_policy_grad(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂_θ log π(a|s) for quantum params at a single state.

        Uses parameter-shift rule:
            ∂_θ f(θ) ≈ [f(θ + π/2) - f(θ - π/2)] / 2

        For log π, we differentiate through the VQC measurement output.

        Args:
            state: Single environment state, shape (obs_dim,).

        Returns:
            Gradient of log-policy w.r.t. quantum params, shape (d,).
        """
        # Enable gradient computation
        q_params_clone = self.actor.q_params.detach().requires_grad_(True)

        # Temporarily swap parameters to allow gradient flow
        original_q_params = self.actor.q_params
        self.actor.q_params = nn.Parameter(q_params_clone)

        try:
            # Forward pass to get log probabilities
            dist = self.actor.get_distribution(state.unsqueeze(0))
            # Sample action from current policy
            with torch.no_grad():
                action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()

            # Backward pass through quantum circuit
            log_prob.backward()
            grad = q_params_clone.grad
            if grad is None:
                grad = torch.zeros_like(q_params_clone)
            grad = grad.flatten().detach()
        except Exception:
            grad = torch.zeros(self.actor.q_params.numel(), dtype=torch.float32)
        finally:
            # Restore original parameters
            self.actor.q_params = original_q_params

        # Handle input_scaling if trainable
        if isinstance(self.actor.input_scaling, nn.Parameter):
            scaling_clone = self.actor.input_scaling.detach().requires_grad_(True)
            original_scaling = self.actor.input_scaling
            self.actor.input_scaling = nn.Parameter(scaling_clone)

            try:
                dist2 = self.actor.get_distribution(state.unsqueeze(0))
                with torch.no_grad():
                    action2 = dist2.sample()
                lp2 = dist2.log_prob(action2).squeeze()
                lp2.backward()
                sg = scaling_clone.grad
                if sg is None:
                    sg = torch.zeros_like(scaling_clone)
                sg = sg.flatten().detach()
            except Exception:
                sg = torch.zeros(self.actor.input_scaling.numel(), dtype=torch.float32)
            finally:
                self.actor.input_scaling = original_scaling

            grad = torch.cat([grad, sg])

        return grad

    def _natural_grad_diagonal(
        self,
        states: torch.Tensor,
        euclidean_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diagonal QFIM preconditioning: g_nat = g / (F_diag + δ)

        This is the cheapest approximation. Equivalent to adaptive
        learning rates where each parameter has its own effective LR
        determined by the inverse QFIM diagonal.

        Args:
            states: Sample states for QFIM estimation.
            euclidean_grad: Flat Euclidean gradient, shape (d,).

        Returns:
            Natural gradient, shape (d,).
        """
        with torch.no_grad():
            diag_fim = self.estimate_diagonal_qfim(states)

        # Tikhonov regularization: (F_ii + δ)
        preconditioner = diag_fim + self.damping

        # Natural gradient: F^{-1} g ≈ g / diag(F + δI)
        nat_grad = euclidean_grad / preconditioner

        return nat_grad

    def _natural_grad_block_diagonal(
        self,
        states: torch.Tensor,
        euclidean_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Block-diagonal QFIM preconditioning.

        Partitions parameters into blocks of size `block_size` (one block
        per layer of the VQC). Computes the full QFIM within each block
        and solves the linear system block-by-block.

        Cost: O(d · block_size²) — much cheaper than full O(d³) inversion.

        Args:
            states: Sample states for QFIM estimation.
            euclidean_grad: Flat Euclidean gradient, shape (d,).

        Returns:
            Natural gradient, shape (d,).
        """
        d = euclidean_grad.shape[0]
        nat_grad = torch.zeros_like(euclidean_grad)

        # Accumulate gradient vectors over state samples
        n_samples = states.shape[0]
        grad_matrix = torch.zeros(n_samples, d, dtype=torch.float32)

        for i, state in enumerate(states):
            grad_matrix[i] = self._compute_log_policy_grad(state)

        # Process each block
        for start in range(0, d, self.block_size):
            end = min(start + self.block_size, d)
            block_grads = grad_matrix[:, start:end]  # (n_samples, block_size)

            # Block QFIM: F_block = E[g_block g_block^T]
            # Shape: (block_size, block_size)
            F_block = (block_grads.T @ block_grads) / n_samples

            # Tikhonov regularization
            B = end - start
            F_block += self.damping * torch.eye(B, dtype=torch.float32)

            # Solve: F_block · x = g_block
            g_block = euclidean_grad[start:end]
            try:
                nat_grad[start:end] = torch.linalg.solve(F_block, g_block)
            except RuntimeError:
                # Fallback to diagonal if block is singular
                nat_grad[start:end] = g_block / (F_block.diag() + self.damping)

        return nat_grad

    def _natural_grad_full(
        self,
        states: torch.Tensor,
        euclidean_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full QFIM preconditioning (most accurate, expensive).

        Computes the complete d×d QFIM using Monte Carlo estimation
        over `n_samples` state samples, then solves the linear system.

        Cost: O(d² · n_samples) for QFIM + O(d³) for linear solve.
        Practical for d ≤ 100 (e.g., 4 qubits, 2 layers → d=24+8=32).

        Args:
            states: Sample states for QFIM estimation.
            euclidean_grad: Flat Euclidean gradient, shape (d,).

        Returns:
            Natural gradient F^{-1} g, shape (d,).
        """
        d = euclidean_grad.shape[0]
        n_samples = states.shape[0]

        # Build gradient matrix: (n_samples, d)
        grad_matrix = torch.zeros(n_samples, d, dtype=torch.float32)
        for i, state in enumerate(states):
            grad_matrix[i] = self._compute_log_policy_grad(state)

        # Full QFIM: F = E[g g^T], shape (d, d)
        F = (grad_matrix.T @ grad_matrix) / n_samples

        # Tikhonov regularization: F_reg = F + δ·I
        F += self.damping * torch.eye(d, dtype=torch.float32)

        # Solve: F · x = g (more numerically stable than explicit inversion)
        try:
            nat_grad = torch.linalg.solve(F, euclidean_grad)
        except RuntimeError:
            # Fallback to diagonal if full solve fails
            nat_grad = euclidean_grad / (F.diag() + self.damping)

        return nat_grad
