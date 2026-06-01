"""
plot_comparison.py — Quantum vs Classical PPO Comparison
=========================================================
Extracts episode reward data from TensorBoard logs and generates
publication-quality comparison charts showing sample efficiency
differences between Quantum PPO and Classical PPO baselines.

Usage:
    python plot_comparison.py
"""

import os
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalar(log_dir: str, tag: str) -> tuple:
    """
    Load a scalar time series from a TensorBoard event file.

    Args:
        log_dir: Path to the TensorBoard log directory.
        tag: Scalar tag name (e.g., 'episode/reward').

    Returns:
        Tuple of (steps, values) as numpy arrays.
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags().get('scalars', []):
        print(f"  Warning: Tag '{tag}' not found in {log_dir}")
        print(f"  Available tags: {ea.Tags().get('scalars', [])}")
        return np.array([]), np.array([])

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute centered moving average with edge handling."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def find_latest_run(runs_dir: str, prefix: str) -> str:
    """Find the most recent TensorBoard run matching a prefix."""
    pattern = os.path.join(runs_dir, f"{prefix}*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No TensorBoard runs found matching '{prefix}*' in {runs_dir}"
        )
    return matches[-1]


def main():
    runs_dir = "runs"

    # ── Find latest runs ────────────────────────────────────────────────
    print("🔍 Looking for TensorBoard runs...")
    quantum_dir = find_latest_run(runs_dir, "hybrid_ppo_")
    classical_dir = find_latest_run(runs_dir, "classical_ppo_")
    print(f"  Quantum:   {quantum_dir}")
    print(f"  Classical: {classical_dir}")

    # ── Load episode rewards ────────────────────────────────────────────
    print("\n📊 Loading episode reward data...")
    q_steps, q_rewards = load_tensorboard_scalar(quantum_dir, "episode/reward")
    c_steps, c_rewards = load_tensorboard_scalar(classical_dir, "episode/reward")

    print(f"  Quantum:   {len(q_rewards)} episodes, "
          f"steps {q_steps[0]:.0f}–{q_steps[-1]:.0f}")
    print(f"  Classical: {len(c_rewards)} episodes, "
          f"steps {c_steps[0]:.0f}–{c_steps[-1]:.0f}")

    # ── Compute smoothed curves ─────────────────────────────────────────
    window = 30
    q_smooth = moving_average(q_rewards, window)
    c_smooth = moving_average(c_rewards, window)

    # ── Find first episode hitting 500 ──────────────────────────────────
    q_first_500_idx = np.where(q_rewards >= 500)[0]
    c_first_500_idx = np.where(c_rewards >= 500)[0]

    q_first_500_step = q_steps[q_first_500_idx[0]] if len(q_first_500_idx) > 0 else None
    c_first_500_step = c_steps[c_first_500_idx[0]] if len(c_first_500_idx) > 0 else None

    q_first_500_ep = q_first_500_idx[0] + 1 if len(q_first_500_idx) > 0 else None
    c_first_500_ep = c_first_500_idx[0] + 1 if len(c_first_500_idx) > 0 else None

    print(f"\n🎯 First episode reaching 500:")
    print(f"  Quantum:   Episode {q_first_500_ep} (step {q_first_500_step})"
          if q_first_500_step else "  Quantum:   Never reached 500")
    print(f"  Classical: Episode {c_first_500_ep} (step {c_first_500_step})"
          if c_first_500_step else "  Classical: Never reached 500")

    # ── Find when smoothed avg exceeds 450 (consistent solve) ───────────
    q_solve_idx = np.where(q_smooth >= 450)[0]
    c_solve_idx = np.where(c_smooth >= 450)[0]

    q_solve_step = q_steps[q_solve_idx[0]] if len(q_solve_idx) > 0 else None
    c_solve_step = c_steps[c_solve_idx[0]] if len(c_solve_idx) > 0 else None

    print(f"\n🏆 Consistent solve (smoothed avg ≥ 450):")
    print(f"  Quantum:   Step {q_solve_step}"
          if q_solve_step else "  Quantum:   Not consistently solved")
    print(f"  Classical: Step {c_solve_step}"
          if c_solve_step else "  Classical: Not consistently solved")

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Episode Reward vs. Environment Steps (Sample Efficiency)
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Quantum PPO vs Classical PPO — Sample Efficiency Comparison\n"
        f"CartPole-v1 | Seed=42 | Quantum: 34 params, Classical: 30 params",
        fontsize=14, fontweight='bold', y=1.02
    )

    # ── Panel 1: Reward vs Steps ────────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(q_steps, q_rewards, alpha=0.08, s=8, c='#6C5CE7', label='_nolegend_')
    ax1.scatter(c_steps, c_rewards, alpha=0.08, s=8, c='#E17055', label='_nolegend_')
    ax1.plot(q_steps, q_smooth, color='#6C5CE7', linewidth=2.5,
             label=f'Quantum PPO (34 params)', zorder=5)
    ax1.plot(c_steps, c_smooth, color='#E17055', linewidth=2.5,
             label=f'Classical PPO (30 params)', zorder=5)

    # Mark first 500
    if q_first_500_step:
        ax1.axvline(q_first_500_step, color='#6C5CE7', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate(f'First 500\n(step {q_first_500_step:,.0f})',
                     xy=(q_first_500_step, 500), fontsize=8, color='#6C5CE7',
                     ha='right', va='bottom',
                     xytext=(-10, -30), textcoords='offset points')
    if c_first_500_step:
        ax1.axvline(c_first_500_step, color='#E17055', linestyle='--', alpha=0.7, linewidth=1)
        ax1.annotate(f'First 500\n(step {c_first_500_step:,.0f})',
                     xy=(c_first_500_step, 500), fontsize=8, color='#E17055',
                     ha='left', va='bottom',
                     xytext=(10, -30), textcoords='offset points')

    ax1.axhline(500, color='#2ECC71', linestyle=':', alpha=0.5, linewidth=1.5,
                label='Max reward (500)')
    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Sample Efficiency: Reward vs. Steps', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.set_xlim(0, 100000)
    ax1.set_ylim(0, 550)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    # ── Panel 2: Reward vs Episodes ─────────────────────────────────────
    ax2 = axes[1]
    q_episodes = np.arange(1, len(q_rewards) + 1)
    c_episodes = np.arange(1, len(c_rewards) + 1)

    ax2.scatter(q_episodes, q_rewards, alpha=0.08, s=8, c='#6C5CE7', label='_nolegend_')
    ax2.scatter(c_episodes, c_rewards, alpha=0.08, s=8, c='#E17055', label='_nolegend_')
    ax2.plot(q_episodes, q_smooth, color='#6C5CE7', linewidth=2.5,
             label=f'Quantum PPO (34 params)', zorder=5)
    ax2.plot(c_episodes, c_smooth, color='#E17055', linewidth=2.5,
             label=f'Classical PPO (30 params)', zorder=5)

    if q_first_500_ep:
        ax2.axvline(q_first_500_ep, color='#6C5CE7', linestyle='--', alpha=0.7, linewidth=1)
    if c_first_500_ep:
        ax2.axvline(c_first_500_ep, color='#E17055', linestyle='--', alpha=0.7, linewidth=1)

    ax2.axhline(500, color='#2ECC71', linestyle=':', alpha=0.5, linewidth=1.5,
                label='Max reward (500)')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Reward', fontsize=12)
    ax2.set_title('Learning Progress: Reward vs. Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.set_ylim(0, 550)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'comparison_quantum_vs_classical.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Saved: {output_path}")
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # Print summary table
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" 📋 COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<35} {'Quantum PPO':>13} {'Classical PPO':>13}")
    print("-" * 65)
    print(f"{'Actor parameters':<35} {'34':>13} {'30':>13}")
    print(f"{'Total episodes (100k steps)':<35} {len(q_rewards):>13} {len(c_rewards):>13}")
    print(f"{'Best reward':<35} {max(q_rewards):>13.0f} {max(c_rewards):>13.0f}")
    print(f"{'Final avg (last 50 eps)':<35} {np.mean(q_rewards[-50:]):>13.1f} {np.mean(c_rewards[-50:]):>13.1f}")

    if q_first_500_step and c_first_500_step:
        print(f"{'First 500 reward (step)':<35} {q_first_500_step:>13,.0f} {c_first_500_step:>13,.0f}")
        print(f"{'First 500 reward (episode)':<35} {q_first_500_ep:>13} {c_first_500_ep:>13}")
        speedup = c_first_500_step / q_first_500_step
        print(f"{'Step speedup (Quantum/Classical)':<35} {speedup:>13.2f}x {'':>13}")

    if q_solve_step and c_solve_step:
        print(f"{'Consistent solve step (avg≥450)':<35} {q_solve_step:>13,.0f} {c_solve_step:>13,.0f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
