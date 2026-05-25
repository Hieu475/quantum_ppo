"""
quantum_actor.py — Generalized Variational Quantum Circuit Actor
=================================================================
Implements the policy network as a parameterized quantum circuit (PQC)
using PennyLane with PyTorch integration. Supports three quantum data
encoding strategies:

  1. Angle Embedding: RY(x_i) per qubit → Variational layers (depth=1 encoding)
  2. Amplitude Embedding: Encode 2^q features into q qubits via amplitudes
  3. Data Re-uploading: Interleave encoding with variational layers at each depth

The classical Pre-encoding NN (state_encoder.py) automatically compresses
any observation space (vector or image) into the required feature dimension,
then normalizes to [-π, π] for quantum gate compatibility.

Architecture:
    Observation → PreEncodingNN(obs → target_dim, Tanh×π) → Quantum Circuit → 
    PauliZ measurements → Linear(q → action_dim) → Softmax → π(a|s)

Key design choices:
  - PreEncodingNN decouples n_qubits from state_dim, enabling any environment
  - Data re-uploading provides non-linearity analogous to activation functions
  - Trainable input scaling adjusts "frequency" of features per layer
  - Rotation gate choice (RX/RY/RZ) is configurable for angle encoding
"""

from typing import Tuple

import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent

import gymnasium as gym

from config import Config
from state_encoder import PreEncodingNN


class QuantumActor(nn.Module):
    """
    Generalized Variational Quantum Circuit (VQC) policy network.

    Supports angle, amplitude, and data re-uploading encoding strategies.
    Automatically adapts to any Gymnasium observation space via the
    PreEncodingNN classical compression layer.

    Attributes:
        n_qubits: Number of qubits in the quantum circuit.
        n_layers: Number of variational ansatz layers.
        encoding_type: Quantum encoding strategy ("angle"|"amplitude"|"data_reuploading").
        pre_encoding_nn: Classical compression network.
        q_params: Trainable quantum rotation angles.
        input_scaling: Trainable input scaling (data_reuploading only).
        output_map: Linear layer mapping measurements to action logits.
    """

    def __init__(self, config: Config, obs_space: gym.spaces.Space) -> None:
        super().__init__()
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.action_dim = config.action_dim
        self.action_type = getattr(config, "action_type", "discrete")
        self.encoding_type = config.encoding_type
        self.rotation_gate = config.rotation_gate

        # ── Classical Pre-encoding NN ────────────────────────────────────
        # Compresses any observation into target_dim features in [-π, π]
        self.pre_encoding_nn = PreEncodingNN(
            obs_space=obs_space,
            n_qubits=config.n_qubits,
            encoding_type=config.encoding_type,
            hidden_dim=config.pre_encoding_hidden,
        )

        # ── PennyLane quantum device (simulator) ────────────────────────
        self.qdev = qml.device("lightning.qubit", wires=self.n_qubits)

        # ── Trainable quantum parameters ────────────────────────────────
        # Shape: (n_layers, n_qubits, 3) for RX, RY, RZ per qubit per layer
        # Initialize with small random values to avoid symmetry and mitigate
        # barren plateaus in shallow circuits.
        init_params = torch.normal(
            mean=0.0,
            std=0.01,
            size=(self.n_layers, self.n_qubits, 3),
            dtype=torch.float32,
        )
        self.q_params = nn.Parameter(init_params)

        # ── Trainable input scaling factors (Data Re-uploading only) ────
        # Shape: (n_layers, n_qubits) — one scale per qubit per layer.
        # Initialized to 1.0 so first forward pass behaves like standard
        # angle encoding; the optimizer then tunes frequencies.
        if self.encoding_type == "data_reuploading":
            self.input_scaling = nn.Parameter(
                torch.ones(self.n_layers, self.n_qubits, dtype=torch.float32)
            )
        else:
            # Register as buffer (not trainable) for angle/amplitude
            self.register_buffer(
                "input_scaling",
                torch.ones(self.n_layers, self.n_qubits, dtype=torch.float32),
            )

        # ── Classical post-processing ───────────────────────────────────
        # Maps n_qubits PauliZ expectations ∈ [-1, 1] → action_dim logits/means
        self.post_nn = nn.Linear(self.n_qubits, self.action_dim)

        # Initialize post-processing mapping with small weights for stable initial policy
        nn.init.xavier_uniform_(self.post_nn.weight, gain=0.1)
        nn.init.zeros_(self.post_nn.bias)

        # ── Continuous Action Space Parameters ──────────────────────────
        if self.action_type == "continuous":
            # Trainable log standard deviation for continuous actions
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            
            # Buffers for action bounds scaling
            action_high = getattr(config, "action_high", None)
            action_low = getattr(config, "action_low", None)
            
            if action_high is not None and action_low is not None:
                self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
                self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
            else:
                self.register_buffer("action_high", torch.ones(self.action_dim, dtype=torch.float32))
                self.register_buffer("action_low", -torch.ones(self.action_dim, dtype=torch.float32))

        # ── Build QNode with PyTorch interface ──────────────────────────
        # Select the appropriate circuit based on encoding_type
        circuit_fn = self._get_circuit_fn()
        self.qnode = qml.QNode(
            circuit_fn,
            self.qdev,
            interface="torch",
            diff_method="adjoint",
        )

    def _get_circuit_fn(self):
        """Return the appropriate quantum circuit function for the encoding type."""
        if self.encoding_type == "angle":
            return self._circuit_angle
        elif self.encoding_type == "amplitude":
            return self._circuit_amplitude
        elif self.encoding_type == "data_reuploading":
            return self._circuit_data_reuploading
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

    # ════════════════════════════════════════════════════════════════════
    # QUANTUM CIRCUITS — Three Encoding Strategies
    # ════════════════════════════════════════════════════════════════════

    def _apply_rotation_gate(self, angle: torch.Tensor, wire: int) -> None:
        """Apply the configured rotation gate (RX, RY, or RZ) to a wire."""
        if self.rotation_gate == "rx":
            qml.RX(angle, wires=wire)
        elif self.rotation_gate == "ry":
            qml.RY(angle, wires=wire)
        elif self.rotation_gate == "rz":
            qml.RZ(angle, wires=wire)

    def _variational_layer(self, weights_layer: torch.Tensor) -> None:
        """
        Apply one variational layer: single-qubit rotations + CNOT entanglement.

        Args:
            weights_layer: Rotation angles, shape (n_qubits, 3).
        """
        # Trainable single-qubit rotations: RX, RY, RZ
        for qubit in range(self.n_qubits):
            qml.RX(weights_layer[qubit, 0], wires=qubit)
            qml.RY(weights_layer[qubit, 1], wires=qubit)
            qml.RZ(weights_layer[qubit, 2], wires=qubit)

        # Entanglement via CNOT chain: (0,1), (1,2), ..., (q-2, q-1)
        for qubit in range(self.n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

    def _circuit_angle(
        self, inputs: torch.Tensor, weights: torch.Tensor,
        scaling: torch.Tensor,
    ) -> list:
        """
        Angle Embedding circuit.

        Encodes each classical feature x_i as a rotation angle on qubit i,
        then applies variational layers for learning.

        |ψ⟩ = ∏_l W_l(θ_l) · ⊗_i R(x_i) |0⟩^⊗q

        Args:
            inputs: Encoded features, shape (n_qubits,).
            weights: Variational params, shape (n_layers, n_qubits, 3).
            scaling: Input scaling (unused for angle, kept for interface).

        Returns:
            List of PauliZ expectation values.
        """
        # Single encoding layer at the beginning
        for qubit in range(self.n_qubits):
            self._apply_rotation_gate(inputs[qubit], wire=qubit)

        # Variational layers
        for layer in range(weights.shape[0]):
            self._variational_layer(weights[layer])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _circuit_amplitude(
        self, inputs: torch.Tensor, weights: torch.Tensor,
        scaling: torch.Tensor,
    ) -> list:
        """
        Amplitude Embedding circuit.

        Encodes 2^q classical features into the probability amplitudes
        of q qubits, then applies variational layers.

        |ψ_x⟩ = Σ_i x_i |i⟩  (with normalization ||x||=1)

        Args:
            inputs: Feature vector, shape (2^n_qubits,).
            weights: Variational params, shape (n_layers, n_qubits, 3).
            scaling: Input scaling (unused, kept for interface).

        Returns:
            List of PauliZ expectation values.
        """
        # Amplitude embedding with automatic L2 normalization
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=range(self.n_qubits),
            normalize=True,
        )

        # Variational layers
        for layer in range(weights.shape[0]):
            self._variational_layer(weights[layer])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def _circuit_data_reuploading(
        self, inputs: torch.Tensor, weights: torch.Tensor,
        scaling: torch.Tensor,
    ) -> list:
        """
        Data Re-uploading circuit.

        Re-encodes the input features at every variational layer,
        interleaving encoding with trainable rotations. This enables
        the circuit to approximate arbitrary functions (Universal
        Function Approximator).

        U(θ,x) = ∏_l W_l(θ_l) · S(w_l · x) |0⟩^⊗q

        Args:
            inputs: Encoded features, shape (n_qubits,).
            weights: Variational params, shape (n_layers, n_qubits, 3).
            scaling: Trainable input scaling, shape (n_layers, n_qubits).

        Returns:
            List of PauliZ expectation values.
        """
        for layer in range(weights.shape[0]):
            # Re-upload: encode scaled input features at every layer
            for qubit in range(self.n_qubits):
                qml.RY(scaling[layer, qubit] * inputs[qubit], wires=qubit)

            # Trainable single-qubit rotations: RX, RY, RZ
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)

            # Entanglement via CNOT chain
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    # ════════════════════════════════════════════════════════════════════
    # FORWARD PASS & DISTRIBUTION METHODS
    # ════════════════════════════════════════════════════════════════════

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: raw observation → action logits.

        Pipeline: state → PreEncodingNN(compress + Tanh×π) → Quantum Circuit → output_map

        Args:
            state: Input state tensor, shape (state_dim,) or (batch, state_dim).
                   For images: (H, W, C) or (batch, H, W, C).

        Returns:
            Discrete: Action logits, shape (action_dim,) or (batch, action_dim).
            Continuous: Action means (scaled), shape (action_dim,) or (batch, action_dim).
        """
        # ── Classical pre-encoding: compress + normalize to [-π, π] ─────
        encoded = self.pre_encoding_nn(state)

        if encoded.dim() == 1:
            # Single state: run circuit once
            measurements = self.qnode(encoded, self.q_params, self.input_scaling)
            # Stack PennyLane outputs into a single tensor
            meas_tensor = torch.stack(list(measurements)).float()
            
            if self.action_type == "discrete":
                return self.post_nn(meas_tensor)
            else:
                raw_mean = torch.tanh(self.post_nn(meas_tensor))
                # Scale from [-1, 1] to [action_low, action_high]
                scale = (self.action_high - self.action_low) / 2.0
                offset = (self.action_high + self.action_low) / 2.0
                return raw_mean * scale + offset
        else:
            # Batched states: use native PennyLane broadcasting instead of vmap
            # (vmap crashes during backward pass with adjoint diff_method)
            if self.encoding_type == "amplitude":
                # AmplitudeEmbedding natively broadcasts (batch, features)
                inputs = encoded
            else:
                # angle and data_reuploading expect inputs[qubit] to be (batch,)
                # so we transpose (batch, n_qubits) -> (n_qubits, batch)
                inputs = encoded.T
                
            measurements = self.qnode(inputs, self.q_params, self.input_scaling)
            batch_tensor = torch.stack(list(measurements), dim=1).float()
            
            if self.action_type == "discrete":
                return self.post_nn(batch_tensor)
            else:
                raw_mean = torch.tanh(self.post_nn(batch_tensor))
                scale = (self.action_high - self.action_low) / 2.0
                offset = (self.action_high + self.action_low) / 2.0
                return raw_mean * scale + offset

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """
        Get the categorical action distribution for a given state.

        Args:
            state: Input state, shape (state_dim,) or (batch, state_dim).

        Returns:
            Categorical or Independent(Normal) distribution over actions.
        """
        output = self.forward(state)
        if self.action_type == "discrete":
            return Categorical(logits=output)
        else:
            # Expand log_std to match batch size if necessary
            std = torch.exp(self.log_std)
            if output.dim() > 1:
                std = std.expand_as(output)
            return Independent(Normal(output, std), 1)

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
