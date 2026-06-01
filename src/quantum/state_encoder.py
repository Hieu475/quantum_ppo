"""
state_encoder.py — Classical Pre-encoding Neural Network
==========================================================
Provides automatic observation-space-aware classical pre-processing to
compress arbitrary Gymnasium observations into a fixed-size feature vector
suitable for quantum circuit encoding.

Architecture selection logic:
  - 1D vector (Box, flat): MLP compression  D → hidden → target_dim
  - 2D/3D image (Box, spatial): CNN feature extractor → MLP head → target_dim
  - If obs_dim == target_dim: Identity pass-through (no compression needed)

The output is always normalized to [-π, π] via Tanh × π, ensuring
compatibility with quantum rotation gates (RX, RY, RZ) that expect
radian-valued inputs with period 2π.

Supports all three quantum encoding strategies:
  - angle / data_reuploading: target_dim = n_qubits
  - amplitude: target_dim = 2^n_qubits
"""

import math
from typing import Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class PreEncodingNN(nn.Module):
    """
    Classical pre-encoding network that automatically adapts to the
    observation space structure and compresses features for quantum encoding.

    The network auto-detects whether the observation is a flat vector or an
    image, builds the appropriate architecture (MLP or CNN), and normalizes
    the output to [-π, π] for quantum gate compatibility.

    Attributes:
        target_dim: Output dimension (n_qubits or 2^n_qubits).
        obs_type: Detected observation type ("vector" or "image").
        net: The compression network (MLP, CNN, or Identity).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        n_qubits: int,
        encoding_type: str = "angle",
        hidden_dim: int = 64,
    ) -> None:
        """
        Initialize the pre-encoding network.

        Args:
            obs_space: Gymnasium observation space.
            n_qubits: Number of qubits in the quantum circuit.
            encoding_type: One of "angle", "amplitude", "data_reuploading".
            hidden_dim: Hidden layer width for MLP compression.
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.hidden_dim = hidden_dim

        # Determine target output dimension based on encoding strategy
        if encoding_type == "amplitude":
            self.target_dim = 2 ** n_qubits
        else:
            # angle and data_reuploading: 1 feature per qubit
            self.target_dim = n_qubits

        # Auto-detect observation type and build network
        self.obs_type, self.net = self._build_network(obs_space)

        # Apply weight initialization
        self._init_weights()

    def _build_network(
        self, obs_space: gym.spaces.Space
    ) -> Tuple[str, nn.Module]:
        """
        Auto-detect observation space type and build the appropriate network.

        Args:
            obs_space: Gymnasium observation space.

        Returns:
            Tuple of (obs_type, network_module).

        Raises:
            ValueError: If the observation space type is not supported.
        """
        if isinstance(obs_space, gym.spaces.Discrete):
            return "discrete", nn.Sequential(
                nn.Embedding(obs_space.n, self.hidden_dim),
                nn.Linear(self.hidden_dim, self.target_dim)
            )

        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(
                f"Unsupported observation space type: {type(obs_space).__name__}. "
                f"Supported types are gym.spaces.Box and gym.spaces.Discrete."
            )

        shape = obs_space.shape

        if len(shape) == 1:
            # ── Flat vector observation (e.g., CartPole, LunarLander) ────
            obs_dim = shape[0]
            return "vector", self._build_mlp(obs_dim)

        elif len(shape) in (2, 3):
            # ── Image observation (e.g., Atari, CarRacing) ───────────────
            return "image", self._build_cnn(shape)

        else:
            raise ValueError(
                f"Unsupported observation shape: {shape}. "
                f"Expected 1D (vector), 2D (grayscale image), or 3D (color image)."
            )

    def _build_mlp(self, obs_dim: int) -> nn.Module:
        """
        Build MLP for flat vector observations.

        If obs_dim == target_dim, returns an Identity module (no compression).
        Otherwise, builds a 2-layer MLP: Linear(D, hidden) → ReLU → Linear(hidden, target).

        Args:
            obs_dim: Input observation dimension.

        Returns:
            MLP network module.
        """
        if obs_dim == self.target_dim:
            # No compression needed — pass through directly
            return nn.Identity()

        return nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

    def _build_cnn(self, obs_shape: Tuple[int, ...]) -> nn.Module:
        """
        Build CNN for image observations (Nature DQN architecture).

        Architecture:
            Conv2d(C, 32, 8, stride=4) → ReLU
            Conv2d(32, 64, 4, stride=2) → ReLU
            Conv2d(64, 64, 3, stride=1) → ReLU
            Flatten → Linear(flat_dim, hidden) → ReLU → Linear(hidden, target)

        Args:
            obs_shape: Observation shape, e.g. (H, W) or (H, W, C) or (C, H, W).

        Returns:
            CNN network module.
        """
        # Handle channel dimension — Gymnasium typically uses (H, W, C)
        if len(obs_shape) == 2:
            # Grayscale without channel dim: (H, W) → treat as 1 channel
            in_channels = 1
            h, w = obs_shape
        elif len(obs_shape) == 3:
            # Check if channels-last (H, W, C) or channels-first (C, H, W)
            if obs_shape[2] <= 4:
                # Likely (H, W, C) — e.g. (210, 160, 3)
                h, w = obs_shape[0], obs_shape[1]
                in_channels = obs_shape[2]
            else:
                # Likely (C, H, W) — e.g. (3, 210, 160)
                in_channels = obs_shape[0]
                h, w = obs_shape[1], obs_shape[2]
        else:
            raise ValueError(f"Cannot build CNN for shape {obs_shape}")

        # Store info for channel reordering in forward pass
        self._img_shape = obs_shape
        self._in_channels = in_channels

        # Nature DQN CNN architecture (Mnih et al., 2015)
        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened dimension by running a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            flat_dim = cnn(dummy).shape[1]

        # MLP head after CNN
        head = nn.Sequential(
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.target_dim),
        )

        return nn.Sequential(cnn, head)

    def _init_weights(self) -> None:
        """Apply orthogonal initialization to all linear layers."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image observations for CNN input.

        Handles:
          - Channel reordering from (B, H, W, C) to (B, C, H, W)
          - Adding batch/channel dimensions as needed
          - Normalizing pixel values from [0, 255] to [0, 1]

        Args:
            x: Raw image tensor.

        Returns:
            Preprocessed image tensor in (B, C, H, W) format.
        """
        # Ensure batch dimension
        if x.dim() == len(self._img_shape):
            x = x.unsqueeze(0)

        # Normalize pixel values to [0, 1] if needed
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Handle (B, H, W) → (B, 1, H, W) for grayscale
        if len(self._img_shape) == 2:
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dim

        # Handle (B, H, W, C) → (B, C, H, W) channel reordering
        elif len(self._img_shape) == 3 and self._img_shape[2] <= 4:
            # Channels-last to channels-first
            x = x.permute(0, 3, 1, 2)

        return x.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: observation → normalized feature vector.

        For angle/data_reuploading encodings, output is clamped to [-π, π] 
        via Tanh × π for quantum rotation gates. For amplitude encoding, 
        raw features are returned.

        Args:
            x: Raw observation tensor (any supported shape).

        Returns:
            Normalized feature vector, shape (..., target_dim).
        """
        # Preprocess inputs
        if self.obs_type == "image":
            x = self._preprocess_image(x)
        elif self.obs_type == "discrete":
            if x.dtype != torch.long:
                x = x.long()
        else:
            if x.dtype != torch.float32:
                x = x.float()

        # Run through compression network
        features = self.net(x)

        # Normalize to [-π, π] for quantum gates
        if self.encoding_type in ["angle", "data_reuploading"]:
            # Tanh maps to [-1, 1], then scale by π
            features = torch.tanh(features) * math.pi

        return features

    @property
    def is_identity(self) -> bool:
        """Check if the pre-encoding network is a simple pass-through."""
        return isinstance(self.net, nn.Identity)
