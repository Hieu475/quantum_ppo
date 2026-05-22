"""
quantum_actor.py — Variational Quantum Circuit Actor
=====================================================
Implements the policy network as a parameterized quantum circuit (PQC)
using PennyLane with PyTorch integration. The circuit uses angle encoding
to embed classical state features, followed by a variational ansatz with
trainable rotations and CNOT entanglement.

Architecture:
    State (4-dim) → RY Encoding → [RX, RY, RZ rotations + CNOT chain] × L layers
    → PauliZ measurements → Linear(4→2) → Softmax → π(a|s)

Key design choice: parameter-shift differentiation ensures hardware-compatible
gradients that can run on real quantum processors (NISQ devices).
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
    Variational Quantum Circuit (VQC) policy network.

    The quantum circuit acts as a feature extractor, producing 4 expectation
    values that are linearly mapped to 2 action logits. The full pipeline
    is differentiable via the parameter-shift rule.

    Attributes:
        n_qubits: Number of qubits (= state dimension).
        n_layers: Number of variational ansatz layers.
        q_params: Trainable quantum rotation angles, shape (n_layers, n_qubits, 3).
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
            diff_method="parameter-shift",
        )

    def _circuit(
        self, inputs: torch.Tensor, weights: torch.Tensor
    ) -> list:
        """
        Define the quantum circuit.

        This function is called by the QNode. It:
        1. Encodes classical state features via RY angle encoding
        2. Applies variational layers (RX, RY, RZ + CNOT entanglement)
        3. Measures PauliZ expectation on each qubit

        Args:
            inputs: State features, shape (n_qubits,).
            weights: Variational parameters, shape (n_layers, n_qubits, 3).

        Returns:
            List of PauliZ expectation values, one per qubit.
        """
        # ── Step 1: Angle Encoding ──────────────────────────────────────
        # Embed each state feature as a rotation angle on its corresponding qubit.
        # RY gates map real values to quantum amplitudes.
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # ── Step 2: Variational Ansatz ──────────────────────────────────
        for layer in range(weights.shape[0]):
            # Trainable single-qubit rotations: RX, RY, RZ
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)

            # Entanglement via CNOT chain: (0,1), (1,2), (2,3)
            # Creates nearest-neighbor correlations between qubits
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # ── Step 3: Measurement ─────────────────────────────────────────
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
            measurements = self.qnode(state, self.q_params)
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
                measurements = self.qnode(state[i], self.q_params)
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
