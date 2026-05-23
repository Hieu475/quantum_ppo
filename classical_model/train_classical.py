"""
train_classical.py — Classical PPO Baseline Training Loop
==========================================================
Mirrors the hybrid quantum PPO training loop (train.py) but uses
a pure classical agent for baseline comparison.

Differences from train.py:
    - Uses ClassicalAgent (tiny MLP actor) instead of HybridAgent
    - No barren plateau diagnostics (not applicable)
    - Logs under 'classical_ppo_*' tag for TensorBoard comparison
    - Same hyperparameters, same seed, same environment

Usage:
    python train_classical.py
    python train_classical.py --total_timesteps 100000 --seed 42
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from classical_agent import ClassicalAgent
from buffer import RolloutBuffer
from utils import (
    set_seed,
    make_env,
    diagnose_entropy,
    compute_grad_norm,
)


# ── Lightweight PPO for ClassicalAgent ──────────────────────────────────
# Reuse the same PPO logic but with ClassicalAgent type
class ClassicalPPO:
    """
    PPO-Clip optimizer for the classical baseline agent.

    Identical algorithm to ppo.py but operates on ClassicalAgent.
    Separated to avoid import dependencies on quantum modules.
    """

    def __init__(self, agent: ClassicalAgent, config: Config) -> None:
        self.agent = agent
        self.config = config

        self.actor_optimizer = torch.optim.Adam(
            agent.get_actor_params(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            agent.get_critic_params(), lr=config.critic_lr
        )

    def update(self, buffer: RolloutBuffer) -> dict:
        """Perform PPO update — same algorithm as ppo.py."""
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_minibatches():
                new_log_probs, entropy, values = self.agent.evaluate_actions(
                    batch.states, batch.actions
                )

                log_ratio = new_log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                advantages = batch.advantages
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = torch.nn.functional.mse_loss(values, batch.returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy_loss
                )

                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                    .float()
                    .mean()
                    .item()
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.get_actor_params(),
                    self.config.max_grad_norm,
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.get_critic_params(),
                    self.config.max_grad_norm,
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_advantage += advantages.mean().item()
                total_actor_grad_norm += (
                    actor_grad_norm.item()
                    if isinstance(actor_grad_norm, torch.Tensor)
                    else actor_grad_norm
                )
                total_critic_grad_norm += (
                    critic_grad_norm.item()
                    if isinstance(critic_grad_norm, torch.Tensor)
                    else critic_grad_norm
                )
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "avg_advantage": total_advantage / n,
            "actor_grad_norm": total_actor_grad_norm / n,
            "critic_grad_norm": total_critic_grad_norm / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_fraction / n,
        }


def train_classical(config: Config) -> None:
    """
    Training function for Classical PPO baseline.

    Identical structure to train() in train.py, but:
    - Uses ClassicalAgent (tiny MLP actor, ~30 params)
    - No quantum diagnostics
    - Logs under 'classical_ppo_*' TensorBoard tag
    """
    # ── Setup ───────────────────────────────────────────────────────────
    set_seed(config.seed)
    env = make_env(config.env_name, config.seed)
    agent = ClassicalAgent(config)
    buffer = RolloutBuffer(config)
    ppo = ClassicalPPO(agent, config)

    # ── TensorBoard ─────────────────────────────────────────────────────
    log_path = os.path.join(
        config.log_dir,
        f"classical_ppo_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    writer = SummaryWriter(log_dir=log_path)
    print(f" TensorBoard logs: {log_path}")
    print(f"   Run: tensorboard --logdir {config.log_dir}")

    # ── Checkpoint directory ────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training state ──────────────────────────────────────────────────
    global_step = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    episode_rewards_history: list = []
    update_count = 0

    state, _ = env.reset(seed=config.seed)
    start_time = time.time()

    # Log configuration
    print("\n" + "=" * 60)
    print(" Classical PPO Baseline Training")
    print("=" * 60)
    print(f"  Environment:     {config.env_name}")
    print(f"  Actor LR:        {config.actor_lr}")
    print(f"  Critic LR:       {config.critic_lr}")
    print(f"  Rollout Steps:   {config.rollout_steps}")
    print(f"  PPO Epochs:      {config.ppo_epochs}")
    print(f"  Minibatch Size:  {config.mini_batch_size}")
    print(f"  Total Timesteps: {config.total_timesteps}")
    print(f"  Device:          {config.device}")
    print(f"  Seed:            {config.seed}")
    print("=" * 60 + "\n")

    # Count parameters
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f" Actor params:  {actor_params} (all classical)")
    print(f" Critic params: {critic_params}")
    print()

    # ════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ════════════════════════════════════════════════════════════════════
    while global_step < config.total_timesteps:
        # ── Phase 1: Rollout Collection ─────────────────────────────────
        buffer.clear()
        rollout_start = time.time()

        for step in range(config.rollout_steps):
            action, log_prob, value = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.store(state, action, log_prob, reward, done, value)

            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                episode_count += 1
                episode_rewards_history.append(episode_reward)

                writer.add_scalar(
                    "episode/reward", episode_reward, global_step
                )
                writer.add_scalar(
                    "episode/length", episode_length, global_step
                )

                if len(episode_rewards_history) >= 100:
                    avg_reward = np.mean(episode_rewards_history[-100:])
                    writer.add_scalar(
                        "episode/avg_reward_100", avg_reward, global_step
                    )

                if episode_count % config.log_interval == 0:
                    avg_recent = np.mean(
                        episode_rewards_history[
                            -min(10, len(episode_rewards_history)) :
                        ]
                    )
                    elapsed = time.time() - start_time
                    fps = global_step / max(elapsed, 1e-8)
                    print(
                        f"Episode {episode_count:5d} | "
                        f"Step {global_step:7d}/{config.total_timesteps} | "
                        f"Reward: {episode_reward:6.1f} | "
                        f"Avg(10): {avg_recent:6.1f} | "
                        f"FPS: {fps:.0f}"
                    )

                episode_reward = 0.0
                episode_length = 0
                state, _ = env.reset()
            else:
                state = next_state

            if global_step >= config.total_timesteps:
                break

        rollout_time = time.time() - rollout_start

        # ── Phase 2: Compute GAE Advantages ─────────────────────────────
        last_value = agent.get_value(state)
        buffer.compute_gae(last_value)

        # ── Phase 3: PPO Update ─────────────────────────────────────────
        update_start = time.time()
        metrics = ppo.update(buffer)
        update_time = time.time() - update_start
        update_count += 1

        # ── Phase 4: Logging ────────────────────────────────────────────
        writer.add_scalar("loss/actor", metrics["actor_loss"], global_step)
        writer.add_scalar("loss/critic", metrics["critic_loss"], global_step)
        writer.add_scalar("loss/entropy", metrics["entropy"], global_step)
        writer.add_scalar(
            "training/avg_advantage", metrics["avg_advantage"], global_step
        )
        writer.add_scalar(
            "training/actor_grad_norm",
            metrics["actor_grad_norm"],
            global_step,
        )
        writer.add_scalar(
            "training/critic_grad_norm",
            metrics["critic_grad_norm"],
            global_step,
        )
        writer.add_scalar(
            "training/approx_kl", metrics["approx_kl"], global_step
        )
        writer.add_scalar(
            "training/clip_fraction", metrics["clip_fraction"], global_step
        )
        writer.add_scalar(
            "timing/rollout_sec", rollout_time, global_step
        )
        writer.add_scalar(
            "timing/update_sec", update_time, global_step
        )

        # ── Policy variance ─────────────────────────────────────────────
        with torch.no_grad():
            sample_state = torch.tensor(
                env.observation_space.sample(),
                dtype=torch.float32,
                device=config.device,
            )
            dist = agent.actor.get_distribution(sample_state)
            policy_probs = dist.probs
            writer.add_scalar(
                "training/policy_variance",
                policy_probs.var().item(),
                global_step,
            )

        # ── Entropy diagnostic (no barren plateau check) ────────────────
        if update_count % config.diagnose_interval == 0:
            entropy_msg = diagnose_entropy(
                metrics["entropy"], config.action_dim
            )
            print(f"  📊 Entropy Diagnostic: {entropy_msg}")

        # ── Phase 5: Checkpointing ──────────────────────────────────────
        if (
            episode_count > 0
            and episode_count % config.save_interval == 0
        ):
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"classical_ppo_step_{global_step}.pt",
            )
            torch.save(
                {
                    "global_step": global_step,
                    "episode_count": episode_count,
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic_state_dict": agent.critic.state_dict(),
                    "actor_optimizer": ppo.actor_optimizer.state_dict(),
                    "critic_optimizer": ppo.critic_optimizer.state_dict(),
                    "config": config,
                    "episode_rewards": episode_rewards_history,
                },
                checkpoint_path,
            )
            print(f"  💾 Checkpoint saved: {checkpoint_path}")

    # ════════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(" Classical PPO Baseline — Training Complete!")
    print("=" * 60)
    print(f"  Total episodes:  {episode_count}")
    print(f"  Total steps:     {global_step}")
    print(f"  Total time:      {total_time:.1f}s")
    if episode_rewards_history:
        print(f"  Best reward:     {max(episode_rewards_history):.1f}")
        print(
            f"  Final avg (100): "
            f"{np.mean(episode_rewards_history[-min(100, len(episode_rewards_history)):]): .1f}"
        )
    print(f"  TensorBoard:     tensorboard --logdir {config.log_dir}")
    print("=" * 60)

    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "classical_ppo_final.pt")
    torch.save(
        {
            "global_step": global_step,
            "episode_count": episode_count,
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "config": config,
            "episode_rewards": episode_rewards_history,
        },
        final_path,
    )
    print(f" Final model saved: {final_path}")

    writer.close()
    env.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classical PPO Baseline for CartPole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--actor_lr", type=float, default=5e-3,
        help="Learning rate for the classical actor.",
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-3,
        help="Learning rate for the classical critic network.",
    )
    parser.add_argument(
        "--critic_hidden", type=int, default=64,
        help="Hidden layer width for the critic MLP.",
    )
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
    parser.add_argument(
        "--total_timesteps", type=int, default=100_000,
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
    """Build configuration from CLI args and launch classical training."""
    args = parse_args()

    config = Config(
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

    train_classical(config)


if __name__ == "__main__":
    main()
