"""
train.py — Training Loop
==========================
Orchestrates the full PPO training pipeline:
1. Rollout collection (agent interacts with environment)
2. GAE advantage computation
3. PPO minibatch updates
4. TensorBoard logging and diagnostics

The training loop is designed for research use with comprehensive
logging of quantum-specific metrics (barren plateau diagnostics,
parameter-shift gradient norms) alongside standard RL metrics.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Config
from agent import HybridAgent
from buffer import RolloutBuffer
from ppo import PPO
from utils import (
    set_seed,
    make_env,
    diagnose_barren_plateau,
    diagnose_entropy,
    compute_grad_norm,
)


def train(config: Config) -> None:
    """
    Main training function for Hybrid Quantum PPO.

    This function implements the full training loop:
    - Rollout collection with the quantum actor
    - GAE computation
    - PPO minibatch optimization
    - Periodic evaluation and checkpoint saving
    - Comprehensive TensorBoard logging

    Args:
        config: Hyperparameter configuration.
    """
    # ── Setup ───────────────────────────────────────────────────────────
    set_seed(config.seed)
    env = make_env(config.env_name, config.seed)
    agent = HybridAgent(config)
    buffer = RolloutBuffer(config)
    ppo = PPO(agent, config)

    # ── TensorBoard ─────────────────────────────────────────────────────
    log_path = os.path.join(
        config.log_dir,
        f"hybrid_ppo_{time.strftime('%Y%m%d_%H%M%S')}",
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
    print(" Hybrid Quantum PPO Training")
    print("=" * 60)
    print(f"  Environment:     {config.env_name}")
    print(f"  Qubits:          {config.n_qubits}")
    print(f"  VQC Layers:      {config.n_layers}")
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
    quantum_params = agent.actor.q_params.numel() + agent.actor.input_scaling.numel()
    print(f" Actor params:  {actor_params} (quantum: {quantum_params}, "
          f"classical: {actor_params - quantum_params})")
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
            # Agent selects action (no gradient needed)
            action, log_prob, value = agent.select_action(state)

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.store(state, action, log_prob, reward, done, value)

            # Track episode statistics
            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                # ── Episode finished ────────────────────────────────────
                episode_count += 1
                episode_rewards_history.append(episode_reward)

                # TensorBoard: episode-level metrics
                writer.add_scalar(
                    "episode/reward", episode_reward, global_step
                )
                writer.add_scalar(
                    "episode/length", episode_length, global_step
                )

                # Moving average reward (last 100 episodes)
                if len(episode_rewards_history) >= 100:
                    avg_reward = np.mean(episode_rewards_history[-100:])
                    writer.add_scalar(
                        "episode/avg_reward_100", avg_reward, global_step
                    )

                # Console logging
                if episode_count % config.log_interval == 0:
                    avg_recent = np.mean(
                        episode_rewards_history[-min(10, len(episode_rewards_history)):]
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

                # Reset episode tracking
                episode_reward = 0.0
                episode_length = 0
                state, _ = env.reset()
            else:
                state = next_state

            if global_step >= config.total_timesteps:
                break

        rollout_time = time.time() - rollout_start

        # ── Phase 2: Compute GAE Advantages ─────────────────────────────
        # Bootstrap with critic's value estimate of the last state
        last_value = agent.get_value(state)
        buffer.compute_gae(last_value)

        # ── Phase 3: PPO Update ─────────────────────────────────────────
        update_start = time.time()
        metrics = ppo.update(buffer)
        update_time = time.time() - update_start
        update_count += 1

        # ── Phase 4: Logging ────────────────────────────────────────────
        # TensorBoard: training metrics
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

        # ── Policy variance (softmax output distribution) ──────────────
        # High variance → diverse actions; low variance → deterministic
        with torch.no_grad():
            sample_state = torch.tensor(
                env.observation_space.sample(), dtype=torch.float32
            )
            dist = agent.actor.get_distribution(sample_state)
            policy_probs = dist.probs
            writer.add_scalar(
                "training/policy_variance",
                policy_probs.var().item(),
                global_step,
            )

        # ── Phase 5: Diagnostics ────────────────────────────────────────
        if (
            config.diagnose_barren_plateau
            and update_count % config.diagnose_interval == 0
        ):
            # Barren plateau check on quantum actor
            bp_diag = diagnose_barren_plateau(agent.actor)
            print(f"  🔬 Barren Plateau Diagnostic: {bp_diag['message']}")
            writer.add_scalar(
                "diagnostics/quantum_mean_grad",
                bp_diag["mean_grad_magnitude"],
                global_step,
            )
            writer.add_scalar(
                "diagnostics/quantum_max_grad",
                bp_diag["max_grad_magnitude"],
                global_step,
            )
            writer.add_scalar(
                "diagnostics/vanishing_param_ratio",
                bp_diag["vanishing_count"] / max(bp_diag["total_params"], 1),
                global_step,
            )

            # Entropy health check
            entropy_msg = diagnose_entropy(
                metrics["entropy"], config.action_dim
            )
            print(f"   Entropy Diagnostic: {entropy_msg}")

        # ── Phase 6: Checkpointing ──────────────────────────────────────
        if (
            episode_count > 0
            and episode_count % config.save_interval == 0
        ):
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"hybrid_ppo_step_{global_step}.pt",
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
            print(f"   Checkpoint saved: {checkpoint_path}")

    # ════════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ════════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(" Training Complete!")
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
    final_path = os.path.join(config.checkpoint_dir, "hybrid_ppo_final.pt")
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
