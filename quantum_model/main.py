"""
main.py — Entry Point for Hybrid Quantum PPO
==============================================
CLI entry point that supports overriding key hyperparameters via
command-line arguments. Automatically detects state_dim and action_dim
from the specified environment.

Usage:
    python main.py                                              # Default (CartPole)
    python main.py --env_name LunarLander-v3 --n_qubits 4      # Different env
    python main.py --encoding_type angle --n_qubits 4           # Angle encoding
    python main.py --encoding_type amplitude --n_qubits 3       # Amplitude encoding
    python main.py --encoding_type data_reuploading --n_layers 3  # Data re-uploading

For TensorBoard monitoring:
    tensorboard --logdir runs
"""

import argparse
import sys

import gymnasium as gym

from config import Config
from train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameter overrides."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical PPO — Generalized State Encoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Environment ─────────────────────────────────────────────────────
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1",
        help="Gymnasium environment ID.",
    )

    # ── Quantum Actor ───────────────────────────────────────────────────
    parser.add_argument(
        "--n_qubits", type=int, default=4,
        help="Number of qubits in the quantum circuit.",
    )
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of variational ansatz layers in the quantum actor.",
    )
    parser.add_argument(
        "--actor_lr", type=float, default=5e-3,
        help="Learning rate for the quantum actor (parameter-shift).",
    )

    # ── State Encoding ──────────────────────────────────────────────────
    parser.add_argument(
        "--encoding_type", type=str, default="data_reuploading",
        choices=["angle", "amplitude", "data_reuploading"],
        help="Quantum data encoding strategy.",
    )
    parser.add_argument(
        "--pre_encoding_hidden", type=int, default=64,
        help="Hidden dimension of the classical pre-encoding network.",
    )
    parser.add_argument(
        "--rotation_gate", type=str, default="ry",
        choices=["rx", "ry", "rz"],
        help="Rotation gate for angle encoding (RX, RY, or RZ).",
    )

    # ── Classical Critic ────────────────────────────────────────────────
    parser.add_argument(
        "--critic_lr", type=float, default=1e-3,
        help="Learning rate for the classical critic network.",
    )
    parser.add_argument(
        "--critic_hidden", type=int, default=64,
        help="Hidden layer width for the critic MLP.",
    )

    # ── PPO Algorithm ───────────────────────────────────────────────────
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95,
        help="GAE lambda for advantage estimation.",
    )
    parser.add_argument(
        "--clip_epsilon", type=float, default=0.2,
        help="PPO clipping parameter.",
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=0.01,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--value_coeff", type=float, default=0.5,
        help="Value loss coefficient.",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5,
        help="Maximum gradient norm for clipping.",
    )

    # ── Training ────────────────────────────────────────────────────────
    parser.add_argument(
        "--total_timesteps", type=int, default=200_000,
        help="Total environment interaction steps.",
    )
    parser.add_argument(
        "--rollout_steps", type=int, default=2048,
        help="Steps per rollout before PPO update.",
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=64,
        help="Minibatch size for PPO epochs.",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=10,
        help="Number of PPO optimization passes per rollout.",
    )

    # ── Misc ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device.",
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs",
        help="TensorBoard log directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Build configuration from CLI args and launch training."""
    args = parse_args()

    # ── Auto-detect environment dimensions ──────────────────────────────
    temp_env = gym.make(args.env_name)
    obs_space = temp_env.observation_space

    # Detect state_dim
    if isinstance(obs_space, gym.spaces.Box):
        if len(obs_space.shape) == 1:
            state_dim = obs_space.shape[0]
        else:
            # Image: use total pixel count (for config validation only)
            import math
            state_dim = math.prod(obs_space.shape)
    else:
        raise ValueError(f"Unsupported observation space: {type(obs_space).__name__}")

    # Detect action space type and dimensions
    action_space = temp_env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_type = "discrete"
        action_dim = action_space.n
        action_high = None
        action_low = None
    elif isinstance(action_space, gym.spaces.Box):
        action_type = "continuous"
        action_dim = action_space.shape[0]
        action_high = action_space.high.tolist()
        action_low = action_space.low.tolist()
    else:
        raise ValueError(
            f"Unsupported action space: {type(action_space).__name__}. "
            f"Only Discrete and Box action spaces are supported."
        )

    temp_env.close()

    print(f"\n🔍 Auto-detected environment: {args.env_name}")
    print(f"   Observation space: {obs_space}")
    print(f"   Action space:      {action_type.capitalize()}({action_dim})")
    print(f"   Encoding type:     {args.encoding_type}")
    print(f"   Qubits:            {args.n_qubits}")

    # ── Build Config ────────────────────────────────────────────────────
    config = Config(
        env_name=args.env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        action_type=action_type,
        action_high=action_high,
        action_low=action_low,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        actor_lr=args.actor_lr,
        encoding_type=args.encoding_type,
        pre_encoding_hidden=args.pre_encoding_hidden,
        rotation_gate=args.rotation_gate,
        critic_lr=args.critic_lr,
        critic_hidden=args.critic_hidden,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        max_grad_norm=args.max_grad_norm,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        device_str=args.device,
        log_dir=args.log_dir,
    )

    # ── Launch training ─────────────────────────────────────────────────
    train(config)


if __name__ == "__main__":
    main()
