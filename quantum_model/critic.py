"""
critic.py — Classical Critic Network (Generalized)
====================================================
A state-value estimator V(s) that automatically adapts to the observation
space structure. Uses MLP for vector observations and CNN + MLP for image
observations. Orthogonal initialization for training stability.

Architecture (vector):
    State (D-dim) → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(1) → V(s)

Architecture (image):
    Image → CNN(Nature DQN) → Flatten → Linear(hidden) → ReLU → Linear(1) → V(s)
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from config import Config


class Critic(nn.Module):
    """
    Classical critic network for state-value estimation.

    Automatically adapts to vector or image observation spaces.
    For vector inputs, uses a 2-layer MLP.
    For image inputs, uses a Nature DQN CNN followed by an MLP head.

    Attributes:
        obs_type: Detected observation type ("vector" or "image").
        feature_extractor: CNN for images or Identity for vectors.
        network: MLP head for value estimation.
    """

    def __init__(self, config: Config, obs_space: gym.spaces.Space) -> None:
        super().__init__()

        self.obs_type = "vector"  # default

        if isinstance(obs_space, gym.spaces.Box):
            shape = obs_space.shape

            if len(shape) == 1:
                # ── Flat vector observation ──────────────────────────────
                self.obs_type = "vector"
                self.feature_extractor = nn.Identity()
                feature_dim = shape[0]

            elif len(shape) in (2, 3):
                # ── Image observation ────────────────────────────────────
                self.obs_type = "image"
                self.feature_extractor, feature_dim = self._build_cnn(shape)

            else:
                raise ValueError(f"Unsupported observation shape: {shape}")
        else:
            raise ValueError(
                f"Unsupported observation space type: {type(obs_space).__name__}"
            )

        # Store image shape info for preprocessing
        if self.obs_type == "image":
            self._img_shape = shape

        # ── MLP value head ───────────────────────────────────────────────
        self.network = nn.Sequential(
            nn.Linear(feature_dim, config.critic_hidden),
            nn.ReLU(),
            nn.Linear(config.critic_hidden, config.critic_hidden),
            nn.ReLU(),
            nn.Linear(config.critic_hidden, 1),
        )

        # ── Orthogonal initialization ───────────────────────────────────
        self._init_weights()

    def _build_cnn(self, obs_shape):
        """
        Build Nature DQN CNN feature extractor for image observations.

        Args:
            obs_shape: Observation shape, e.g. (H, W) or (H, W, C).

        Returns:
            Tuple of (cnn_module, output_feature_dim).
        """
        if len(obs_shape) == 2:
            in_channels = 1
            h, w = obs_shape
        elif len(obs_shape) == 3:
            if obs_shape[2] <= 4:
                h, w = obs_shape[0], obs_shape[1]
                in_channels = obs_shape[2]
            else:
                in_channels = obs_shape[0]
                h, w = obs_shape[1], obs_shape[2]
        else:
            raise ValueError(f"Cannot build CNN for shape {obs_shape}")

        self._in_channels = in_channels

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            flat_dim = cnn(dummy).shape[1]

        return cnn, flat_dim

    def _preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess image: normalize pixels, reorder channels."""
        if x.dim() == len(self._img_shape):
            x = x.unsqueeze(0)

        if x.max() > 1.0:
            x = x.float() / 255.0

        if len(self._img_shape) == 2:
            if x.dim() == 3:
                x = x.unsqueeze(1)
        elif len(self._img_shape) == 3 and self._img_shape[2] <= 4:
            x = x.permute(0, 3, 1, 2)

        return x.float()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization to linear and conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Last linear layer gets smaller gain for stable initial predictions
        last_linear = [m for m in self.network if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V(s).

        Args:
            state: State tensor. For vectors: shape (state_dim,) or (batch, state_dim).
                   For images: shape (H, W, C) or (batch, H, W, C).

        Returns:
            Value estimate, shape (1,) or (batch, 1).
        """
        if self.obs_type == "image":
            state = self._preprocess_image(state)

        features = self.feature_extractor(state)
        return self.network(features)
