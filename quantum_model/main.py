"""
main.py — Entry Point for Hybrid Quantum PPO
==============================================
CLI entry point that supports overriding key hyperparameters via
command-line arguments. Launches the full training pipeline.

Usage:
    python main.py                           # Default config
    python main.py --total_timesteps 50000   # Quick test run
    python main.py --n_layers 3 --actor_lr 1e-3 --seed 123

For TensorBoard monitoring:
    tensorboard --logdir runs
"""

import argparse
import sys

from config import Config
from train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hyperparameter overrides."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical PPO for CartPole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Quantum Actor ───────────────────────────────────────────────────
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of variational ansatz layers in the quantum actor.",
    )
    parser.add_argument(
        "--actor_lr", type=float, default=5e-3,
        help="Learning rate for the quantum actor (parameter-shift).",
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

    # ── Build Config ────────────────────────────────────────────────────
    config = Config(
        n_layers=args.n_layers,
        actor_lr=args.actor_lr,
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
