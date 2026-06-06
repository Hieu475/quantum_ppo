"""
config.py — Centralized Hyperparameter Configuration
=====================================================
All tunable parameters are stored in a single dataclass for reproducibility
and easy hyperparameter sweeps. Separating actor/critic learning rates is
critical because quantum circuits typically require larger step sizes due
to the bounded nature of parameter-shift gradients.

Supports generalized state encoding with three quantum encoding strategies:
  - "angle": Angle Embedding (1 feature per qubit)
  - "amplitude": Amplitude Embedding (2^q features encoded in q qubits)
  - "data_reuploading": Data Re-uploading (features re-encoded at each layer)
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Config:
    """Complete hyperparameter configuration for Hybrid Quantum PPO."""

    # ── Environment ──────────────────────────────────────────────────────
    env_name: str = "CartPole-v1"
    state_dim: int = 4          # Observation dimension (auto-detected in main)
    action_dim: int = 2         # Action space size (auto-detected in main)
    action_type: str = "discrete" # "discrete" | "continuous" (auto-detected)
    action_high: Optional[list] = None # Upper bound for continuous actions
    action_low: Optional[list] = None  # Lower bound for continuous actions

    # ── Quantum Actor ────────────────────────────────────────────────────
    n_qubits: int = 4           # Number of qubits in the quantum circuit
    n_layers: int = 2           # Number of variational ansatz layers
    actor_lr: float = 5e-3      # Higher LR for quantum params (parameter-shift)

    # ── State Encoding ───────────────────────────────────────────────────
    encoding_type: str = "data_reuploading"  # "angle" | "amplitude" | "data_reuploading"
    pre_encoding_hidden: int = 64            # Hidden dim of Pre-encoding MLP/CNN head
    rotation_gate: str = "ry"                # Rotation gate for angle encoding: "rx" | "ry" | "rz"

    # ── Classical Compression Head (QuantumActor first layer) ─────────────
    # Explicit Linear(obs_dim → n_qubits) bottleneck inside QuantumActor.
    # None  → single Linear layer  : obs_dim → n_qubits  (mặc định)
    # int   → 2-layer MLP          : obs_dim → hidden → n_qubits
    # Ví dụ cho HPC env (obs_dim=21, n_qubits=6): Linear(21, 6) + LayerNorm + Tanh×π
    compression_hidden: Optional[int] = None

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
    total_timesteps: int = 200_000   # Total environment steps
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
    diagnose_barren_plateau: bool = True   # Monitor quantum gradient magnitudes
    diagnose_interval: int = 10            # Run diagnostics every N updates

    @property
    def device(self) -> torch.device:
        """Resolve device string to torch.device."""
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # ── Encoding type validation ─────────────────────────────────────
        valid_encodings = ("angle", "amplitude", "data_reuploading")
        assert self.encoding_type in valid_encodings, (
            f"encoding_type must be one of {valid_encodings}, "
            f"got '{self.encoding_type}'."
        )

        # ── Action type validation ───────────────────────────────────────
        valid_actions = ("discrete", "continuous")
        assert self.action_type in valid_actions, (
            f"action_type must be one of {valid_actions}, "
            f"got '{self.action_type}'."
        )

        # ── Rotation gate validation ─────────────────────────────────────
        valid_gates = ("rx", "ry", "rz")
        assert self.rotation_gate in valid_gates, (
            f"rotation_gate must be one of {valid_gates}, "
            f"got '{self.rotation_gate}'."
        )

        # ── Amplitude encoding dimension check ──────────────────────────
        if self.encoding_type == "amplitude":
            max_features = 2 ** self.n_qubits
            assert self.state_dim <= max_features, (
                f"Amplitude encoding with {self.n_qubits} qubits can encode "
                f"at most {max_features} features, but state_dim={self.state_dim}. "
                f"Increase n_qubits or use a different encoding."
            )

        # ── General validations ──────────────────────────────────────────
        assert self.rollout_steps >= self.mini_batch_size, (
            f"rollout_steps ({self.rollout_steps}) must be >= "
            f"mini_batch_size ({self.mini_batch_size})."
        )
        assert 0 < self.clip_epsilon < 1, "clip_epsilon must be in (0, 1)."
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]."
        assert 0 < self.gae_lambda <= 1, "gae_lambda must be in (0, 1]."
        assert self.n_qubits >= 1, "n_qubits must be at least 1."
        assert self.n_layers >= 1, "n_layers must be at least 1."
