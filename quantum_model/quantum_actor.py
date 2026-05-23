"""
quantum_actor.py — Variational Quantum Circuit Actor with Data Re-uploading
============================================================================
Implements the policy network as a parameterized quantum circuit (PQC)
using PennyLane with PyTorch integration. Uses the Data Re-uploading
technique (Pérez-Salinas et al., 2020) to interleave data encoding with
variational layers, enabling the circuit to express higher-order Fourier
series and act as a universal function approximator.

Architecture:
    [RY(w_scale · s) Encoding → RX, RY, RZ rotations → CNOT chain] × L layers
    → PauliZ measurements → Linear(4→2) → Softmax → π(a|s)

Key design choices:
  - Data re-uploading provides non-linearity analogous to activation functions
    in classical neural networks, allowing the Quantum Actor to learn complex,
    smooth policies.
  - Trainable input scaling factors adjust the "frequency" of input features
    per layer, accelerating convergence.
"""

from typing import Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import Config


class QuantumActor(nn.Module):
    """
    Variational Quantum Circuit (VQC) policy network with Data Re-uploading.

    The quantum circuit acts as a feature extractor, producing 4 expectation
    values that are linearly mapped to 2 action logits. Data Re-uploading
    re-encodes the input state at every variational layer, enabling the
    circuit to represent higher-order Fourier series (non-linear functions).

    Attributes:
        n_qubits: Number of qubits (= state dimension).
        n_layers: Number of variational ansatz layers.
        q_params: Trainable quantum rotation angles, shape (n_layers, n_qubits, 3).
        input_scaling: Trainable scaling factors for re-uploaded inputs,
                       shape (n_layers, n_qubits).
        output_map: Linear layer mapping measurements to action logits.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.action_dim = config.action_dim

        # ── PennyLane quantum device (simulator) ────────────────────────
        self.qdev = qml.device("default.qubit", wires=self.n_qubits)

        # ── Trainable quantum parameters ────────────────────────────────
        # Shape: (n_layers, n_qubits, 3) for RX, RY, RZ per qubit per layer
        # Initialize with small random values to avoid symmetry and mitigate
        # barren plateaus in shallow circuits.
        init_params = np.random.uniform(
            low=-np.pi * 0.1,
            high=np.pi * 0.1,
            size=(self.n_layers, self.n_qubits, 3),
        )
        self.q_params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))

        # ── Trainable input scaling factors (Data Re-uploading) ─────────
        # Shape: (n_layers, n_qubits) — one scale per qubit per layer.
        # Initialized to 1.0 so the first forward pass behaves like
        # standard angle encoding; the optimizer then tunes frequencies.
        self.input_scaling = nn.Parameter(
            torch.ones(self.n_layers, self.n_qubits, dtype=torch.float32)
        )

        # ── Classical post-processing ───────────────────────────────────
        # Maps 4 PauliZ expectations ∈ [-1, 1] → 2 action logits
        self.output_map = nn.Linear(self.n_qubits, self.action_dim)

        # Initialize output mapping with small weights for stable initial policy
        nn.init.xavier_uniform_(self.output_map.weight, gain=0.1)
        nn.init.zeros_(self.output_map.bias)

        # ── Build QNode with PyTorch interface ──────────────────────────
        # diff_method="parameter-shift" ensures hardware-compatible gradients
        # interface="torch" ensures outputs are torch tensors with autograd support
        self.qnode = qml.QNode(
            self._circuit,
            self.qdev,
            interface="torch",
            diff_method="backprop",
        )

    def _circuit(
        self, inputs: torch.Tensor, weights: torch.Tensor,
        scaling: torch.Tensor,
    ) -> list:
        """
        Define the quantum circuit with Data Re-uploading.

        At each variational layer the input state is re-encoded via
        RY(scaling[layer] · inputs) *before* the trainable rotations.
        This interleaving produces a unitary of the form:
            U = ∏_{l=1}^{L} W_l(θ_l) · S(w_l · s)
        enabling the circuit to express higher-order Fourier series.

        Args:
            inputs: State features, shape (n_qubits,).
            weights: Variational parameters, shape (n_layers, n_qubits, 3).
            scaling: Input scaling factors, shape (n_layers, n_qubits).

        Returns:
            List of PauliZ expectation values, one per qubit.
        """
        # ── Data Re-uploading + Variational Ansatz ──────────────────────
        for layer in range(weights.shape[0]):
            # Re-upload: encode scaled input features at every layer
            for qubit in range(self.n_qubits):
                qml.RY(scaling[layer, qubit] * inputs[qubit], wires=qubit)

            # Trainable single-qubit rotations: RX, RY, RZ
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)

            # Entanglement via CNOT chain: (0,1), (1,2), (2,3)
            # Creates nearest-neighbor correlations between qubits
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # ── Measurement ─────────────────────────────────────────────────
        # Expectation values of PauliZ on each qubit ∈ [-1, 1]
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → action logits.

        Handles both single states and batched states.

        Args:
            state: Input state tensor, shape (state_dim,) or (batch, state_dim).

        Returns:
            Action logits, shape (action_dim,) or (batch, action_dim).
        """
        if state.dim() == 1:
            # Single state: run circuit once
            measurements = self.qnode(state, self.q_params, self.input_scaling)
            # Stack PennyLane outputs into a single tensor
            # PennyLane returns float64; cast to float32 for nn.Linear compatibility
            meas_tensor = torch.stack(list(measurements)).float()
            logits = self.output_map(meas_tensor)
            return logits
        else:
            # Batched states: run circuit for each sample
            # Note: PennyLane parameter-shift does not natively vectorize
            # over batch dimensions, so we loop (acceptable for research).
            batch_logits = []
            for i in range(state.shape[0]):
                measurements = self.qnode(state[i], self.q_params, self.input_scaling)
                meas_tensor = torch.stack(list(measurements)).float()
                logits = self.output_map(meas_tensor)
                batch_logits.append(logits)
            return torch.stack(batch_logits)

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
