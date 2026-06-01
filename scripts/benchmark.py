"""
benchmark.py — Comprehensive Benchmarking & Analysis
======================================================
Extracts raw data from TensorBoard event files and generates
publication-quality scientific comparison figures for the
Quantum PPO vs Classical PPO experiment.

Outputs:
  1. benchmark_data/quantum_rewards.csv & classical_rewards.csv — Raw exported data
  2. benchmark_data/quantum_timing.csv & classical_timing.csv   — Timing/FPS data
  3. fig1_reward_convergence.png — Reward convergence with ±σ shaded bands
  4. fig2_stability_analysis.png — Stability: rolling std, catastrophic forgetting check
  5. fig3_computational_cost.png — Wall-clock time & FPS comparison
  6. fig4_full_dashboard.png     — Combined 4-panel research dashboard
  7. benchmark_report.json       — Machine-readable summary statistics

Usage:
    cd /path/to/quantum_ppo
    python benchmark.py

    # Custom runs directory
    python benchmark.py --runs_dir runs --output_dir benchmark_data
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

# For TensorBoard data extraction
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[WARNING] tensorboard not installed. Install with: pip install tensorboard")

# ── Style Configuration ──────────────────────────────────────────────────────
QUANTUM_COLOR   = "#6C5CE7"   # Purple — quantum model
CLASSICAL_COLOR = "#E17055"   # Orange-red — classical model
SOLVE_COLOR     = "#00B894"   # Green — solved threshold
GRID_ALPHA      = 0.25
FONT_FAMILY     = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#FAFAFA",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         GRID_ALPHA,
    "grid.linestyle":     "--",
    "font.family":        FONT_FAMILY,
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "legend.framealpha":  0.9,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
})


# ════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION FROM TENSORBOARD
# ════════════════════════════════════════════════════════════════════════════

def load_scalar(log_dir: str, tag: str):
    """
    Load a scalar tag from a TensorBoard event file.

    Args:
        log_dir: Path to the run directory containing events.out.tfevents.*
        tag: TensorBoard scalar tag, e.g. 'episode/reward'.

    Returns:
        Tuple of (wall_times, steps, values) as numpy arrays, or empty arrays.
    """
    if not HAS_TENSORBOARD:
        return np.array([]), np.array([]), np.array([])

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        return np.array([]), np.array([]), np.array([])

    events = ea.Scalars(tag)
    wall_times = np.array([e.wall_time for e in events])
    steps      = np.array([e.step      for e in events])
    values     = np.array([e.value     for e in events])
    return wall_times, steps, values


def find_run(runs_dir: str, prefix: str) -> str:
    """Find the most recent TensorBoard run directory matching prefix."""
    pattern = os.path.join(runs_dir, f"{prefix}*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No run matching '{prefix}*' found in {runs_dir}. "
            f"Available: {os.listdir(runs_dir)}"
        )
    return matches[-1]


def extract_all_scalars(log_dir: str) -> dict:
    """
    Extract all available scalar tags from a TensorBoard run.

    Returns:
        Dict mapping tag → (wall_times, steps, values).
    """
    if not HAS_TENSORBOARD:
        return {}

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            data[tag] = {
                "wall_times": np.array([e.wall_time for e in events]),
                "steps":      np.array([e.step      for e in events]),
                "values":     np.array([e.value     for e in events]),
            }
    return data


# ════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export_to_csv(data: dict, output_dir: str, prefix: str) -> None:
    """
    Export extracted TensorBoard scalars to CSV files.

    Args:
        data: Dict from extract_all_scalars().
        output_dir: Directory to write CSV files.
        prefix: File prefix (e.g. 'quantum' or 'classical').
    """
    os.makedirs(output_dir, exist_ok=True)

    # One CSV per important tag group
    tag_groups = {
        "rewards": ["episode/reward", "episode/length", "episode/avg_reward_100"],
        "losses":  ["loss/actor", "loss/critic", "loss/entropy"],
        "training":["training/approx_kl", "training/clip_fraction",
                    "training/avg_advantage", "training/policy_variance"],
        "timing":  ["timing/rollout_sec", "timing/update_sec"],
        "diagnostics": [
            "diagnostics/quantum_mean_grad",
            "diagnostics/quantum_max_grad",
            "diagnostics/vanishing_param_ratio",
        ],
    }

    for group_name, tags in tag_groups.items():
        rows = []
        header = ["step", "wall_time"] + [t.split("/")[-1] for t in tags if t in data]

        # Find common steps (use episode/reward steps as anchor if available)
        anchor_tag = next((t for t in tags if t in data), None)
        if anchor_tag is None:
            continue

        for i, step in enumerate(data[anchor_tag]["steps"]):
            row = [int(step), float(data[anchor_tag]["wall_times"][i])]
            for tag in tags:
                if tag in data and i < len(data[tag]["values"]):
                    row.append(float(data[tag]["values"][i]))
            rows.append(row)

        if not rows:
            continue

        csv_path = os.path.join(output_dir, f"{prefix}_{group_name}.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")

        print(f"  Exported: {csv_path} ({len(rows)} rows)")


# ════════════════════════════════════════════════════════════════════════════
# SMOOTHING UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Compute centered moving average with edge-padding."""
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    pad_l = window // 2
    pad_r = window - 1 - pad_l
    padded = np.pad(values, (pad_l, pad_r), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rolling_std(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling standard deviation over a window."""
    result = np.zeros(len(values))
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result[i] = np.std(values[lo:hi])
    return result


def rolling_stats(values: np.ndarray, window: int = 50):
    """
    Compute rolling mean and standard deviation simultaneously.

    Returns:
        Tuple of (mean, std) arrays of the same length as values.
    """
    mean = moving_average(values, window)
    std  = rolling_std(values, window)
    return mean, std


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: REWARD CONVERGENCE WITH ±σ SHADED BANDS
# ════════════════════════════════════════════════════════════════════════════

def plot_reward_convergence(
    q_steps, q_rewards,
    c_steps, c_rewards,
    output_path: str,
    window: int = 50,
) -> None:
    """
    Figure 1: Episode reward vs. timesteps with shaded ±σ stability bands.

    The shaded region represents ±1 rolling std deviation around the
    smoothed mean. A narrowing band toward the end indicates convergence
    and absence of catastrophic forgetting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Figure 1 — Reward Convergence: Quantum PPO vs Classical PPO\n"
        "Shaded bands = ±1σ rolling standard deviation (stability indicator)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, steps, rewards, color, label, params in [
        (axes[0], q_steps, q_rewards, QUANTUM_COLOR,   "Quantum PPO",   "~50 params"),
        (axes[1], c_steps, c_rewards, CLASSICAL_COLOR, "Classical PPO", "~30 params"),
    ]:
        mean, std = rolling_stats(rewards, window=window)

        # Raw scatter (very transparent)
        ax.scatter(steps, rewards, alpha=0.10, s=6, c=color, zorder=1)

        # ±σ shaded band
        ax.fill_between(
            steps, mean - std, mean + std,
            color=color, alpha=0.18, label=f"±1σ band", zorder=2,
        )

        # Smoothed mean line
        ax.plot(steps, mean, color=color, linewidth=2.5,
                label=f"{label} ({params})", zorder=5)

        # Perfect solve line
        ax.axhline(500, color=SOLVE_COLOR, linestyle=":", linewidth=1.5,
                   alpha=0.8, label="Perfect (500)")

        # Mark first time smoothed curve exceeds 450
        solve_idx = np.where(mean >= 450)[0]
        if len(solve_idx) > 0:
            solve_step = steps[solve_idx[0]]
            ax.axvline(solve_step, color=color, linestyle="--", alpha=0.6, linewidth=1.2)
            ax.annotate(
                f"Stable solve\n@{solve_step/1000:.0f}k steps",
                xy=(solve_step, 450),
                xytext=(solve_step + (steps[-1] - steps[0]) * 0.05, 350),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"{label}\nReward Convergence (smoothing window={window})")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 560)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

        # Annotate final performance
        final_mean = np.mean(rewards[-min(100, len(rewards)):])
        final_std  = np.std(rewards[-min(100, len(rewards)):])
        ax.text(
            0.97, 0.05,
            f"Final avg (last 100):\n"
            f"μ = {final_mean:.1f} ± {final_std:.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: STABILITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def plot_stability_analysis(
    q_steps, q_rewards,
    c_steps, c_rewards,
    output_path: str,
    window: int = 50,
) -> None:
    """
    Figure 2: Stability analysis — rolling std over time.

    A narrowing rolling std indicates stable convergence.
    A rising std at the end indicates catastrophic forgetting / instability.
    Also shows reward distribution histograms for the final 20% of training.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Rolling Std over time ──────────────────────────────────
    ax_std = fig.add_subplot(gs[0, :2])

    q_std = rolling_std(q_rewards, window=window)
    c_std = rolling_std(c_rewards, window=window)

    ax_std.plot(q_steps, q_std, color=QUANTUM_COLOR, linewidth=2,
                label="Quantum PPO — rolling σ")
    ax_std.plot(c_steps, c_std, color=CLASSICAL_COLOR, linewidth=2,
                label="Classical PPO — rolling σ")
    ax_std.fill_between(q_steps, 0, q_std, color=QUANTUM_COLOR, alpha=0.12)
    ax_std.fill_between(c_steps, 0, c_std, color=CLASSICAL_COLOR, alpha=0.12)

    ax_std.set_xlabel("Environment Steps")
    ax_std.set_ylabel("Rolling Std σ (reward)")
    ax_std.set_title(
        "Policy Stability — Rolling Standard Deviation over Training\n"
        "(Narrowing toward end = stable convergence; Rising = catastrophic forgetting)"
    )
    ax_std.legend()
    ax_std.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ── Panel B: Reward distribution (last 20% of training) ────────────
    ax_hist = fig.add_subplot(gs[0, 2])
    final_q = q_rewards[int(0.8 * len(q_rewards)):]
    final_c = c_rewards[int(0.8 * len(c_rewards)):]

    bins = np.linspace(0, 520, 30)
    ax_hist.hist(final_q, bins=bins, color=QUANTUM_COLOR, alpha=0.65,
                 label=f"Quantum (μ={np.mean(final_q):.0f})", density=True)
    ax_hist.hist(final_c, bins=bins, color=CLASSICAL_COLOR, alpha=0.65,
                 label=f"Classical (μ={np.mean(final_c):.0f})", density=True)

    ax_hist.axvline(np.mean(final_q), color=QUANTUM_COLOR,
                    linestyle="--", linewidth=1.5, alpha=0.8)
    ax_hist.axvline(np.mean(final_c), color=CLASSICAL_COLOR,
                    linestyle="--", linewidth=1.5, alpha=0.8)

    ax_hist.set_xlabel("Episode Reward")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Reward Distribution\n(Last 20% of training episodes)")
    ax_hist.legend(fontsize=9)

    # ── Panel C: Smoothed reward + ±2σ bands (catastrophic forgetting) ──
    ax_cf = fig.add_subplot(gs[1, :])
    q_mean, q_std_r = rolling_stats(q_rewards, window=window)
    c_mean, c_std_r = rolling_stats(c_rewards, window=window)

    # ±1σ and ±2σ bands
    for sigma, alpha in [(2, 0.08), (1, 0.18)]:
        ax_cf.fill_between(
            q_steps,
            q_mean - sigma * q_std_r,
            q_mean + sigma * q_std_r,
            color=QUANTUM_COLOR, alpha=alpha,
        )
        ax_cf.fill_between(
            c_steps,
            c_mean - sigma * c_std_r,
            c_mean + sigma * c_std_r,
            color=CLASSICAL_COLOR, alpha=alpha,
        )

    ax_cf.plot(q_steps, q_mean, color=QUANTUM_COLOR, linewidth=2.5,
               label="Quantum PPO (mean)")
    ax_cf.plot(c_steps, c_mean, color=CLASSICAL_COLOR, linewidth=2.5,
               label="Classical PPO (mean)")
    ax_cf.axhline(500, color=SOLVE_COLOR, linestyle=":", linewidth=1.5,
                  alpha=0.8, label="Perfect score (500)")

    # Shade final 20% region to highlight end-of-training stability
    q_cutoff = q_steps[int(0.8 * len(q_steps))]
    c_cutoff = c_steps[int(0.8 * len(c_steps))]
    ax_cf.axvspan(max(q_cutoff, c_cutoff), max(q_steps[-1], c_steps[-1]),
                  color="gray", alpha=0.06, label="Final 20% (stability check)")

    # Add legend patches for sigma bands
    q_patch1 = mpatches.Patch(color=QUANTUM_COLOR, alpha=0.25, label="Quantum ±1σ/±2σ")
    c_patch1 = mpatches.Patch(color=CLASSICAL_COLOR, alpha=0.25, label="Classical ±1σ/±2σ")

    handles, labels = ax_cf.get_legend_handles_labels()
    ax_cf.legend(handles + [q_patch1, c_patch1], labels + [q_patch1.get_label(), c_patch1.get_label()],
                 loc="lower right", fontsize=9)

    ax_cf.set_xlabel("Environment Steps")
    ax_cf.set_ylabel("Episode Reward")
    ax_cf.set_title(
        "Figure 2 — Policy Stability: ±1σ and ±2σ Confidence Bands\n"
        "Narrowing bands in the final 20% confirm absence of catastrophic forgetting"
    )
    ax_cf.set_ylim(0, 560)
    ax_cf.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle(
        "Stability Analysis — Quantum PPO vs Classical PPO",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: COMPUTATIONAL COST COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def plot_computational_cost(
    q_data: dict,
    c_data: dict,
    output_path: str,
) -> None:
    """
    Figure 3: Wall-clock time and FPS comparison.

    Compares:
      - Rollout collection time (env interaction)
      - Update/optimization time (VQC gradient vs NN gradient)
      - Effective FPS (Frames Per Second = steps / wall time)
      - Cumulative wall-clock time over training
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Figure 3 — Computational Cost: Quantum PQC vs Classical NN\n"
        "Key metric: Time per 1000 optimization steps (lower = cheaper)",
        fontsize=14, fontweight="bold",
    )

    timing_tags = {
        "rollout": "timing/rollout_sec",
        "update":  "timing/update_sec",
    }

    q_rollout = q_data.get("timing/rollout_sec", {})
    q_update  = q_data.get("timing/update_sec", {})
    c_rollout = c_data.get("timing/rollout_sec", {})
    c_update  = c_data.get("timing/update_sec", {})

    has_timing = all(
        len(d.get("values", [])) > 0
        for d in [q_rollout, q_update, c_rollout, c_update]
    )

    # ── Panel 1: Rollout time per update ────────────────────────────────
    ax1 = axes[0, 0]
    if has_timing:
        ax1.plot(q_rollout["steps"], moving_average(q_rollout["values"], 10),
                 color=QUANTUM_COLOR, linewidth=2, label="Quantum PPO")
        ax1.plot(c_rollout["steps"], moving_average(c_rollout["values"], 10),
                 color=CLASSICAL_COLOR, linewidth=2, label="Classical PPO")
        ax1.set_xlabel("Environment Steps")
        ax1.set_ylabel("Rollout Time (seconds)")
        ax1.set_title("Rollout Collection Time per Update")
        ax1.legend()
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    else:
        _placeholder_panel(ax1, "Rollout Time\n(No timing data in logs)")

    # ── Panel 2: Update (gradient) time per update ───────────────────────
    ax2 = axes[0, 1]
    if has_timing:
        ax2.plot(q_update["steps"], moving_average(q_update["values"], 10),
                 color=QUANTUM_COLOR, linewidth=2, label="Quantum PPO (VQC gradient)")
        ax2.plot(c_update["steps"], moving_average(c_update["values"], 10),
                 color=CLASSICAL_COLOR, linewidth=2, label="Classical PPO (NN backprop)")
        ax2.set_xlabel("Environment Steps")
        ax2.set_ylabel("Gradient Update Time (seconds)")
        ax2.set_title("Gradient Optimization Time per Update\n(VQC adjoint diff vs. standard backprop)")
        ax2.legend()
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    else:
        _placeholder_panel(ax2, "Update Time\n(No timing data in logs)")

    # ── Panel 3: Effective FPS ───────────────────────────────────────────
    ax3 = axes[1, 0]
    if has_timing:
        # FPS = rollout_steps / rollout_time  (rollout_steps ≈ 2048 per update)
        rollout_steps = 2048
        q_fps = rollout_steps / np.maximum(q_rollout["values"], 1e-8)
        c_fps = rollout_steps / np.maximum(c_rollout["values"], 1e-8)

        ax3.plot(q_rollout["steps"], moving_average(q_fps, 10),
                 color=QUANTUM_COLOR, linewidth=2, label=f"Quantum PPO")
        ax3.plot(c_rollout["steps"], moving_average(c_fps, 10),
                 color=CLASSICAL_COLOR, linewidth=2, label=f"Classical PPO")

        q_fps_mean = np.mean(q_fps)
        c_fps_mean = np.mean(c_fps)
        ax3.axhline(q_fps_mean, color=QUANTUM_COLOR, linestyle="--",
                    alpha=0.5, linewidth=1)
        ax3.axhline(c_fps_mean, color=CLASSICAL_COLOR, linestyle="--",
                    alpha=0.5, linewidth=1)

        ax3.text(0.02, q_fps_mean / ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 0.5,
                 f"μ={q_fps_mean:.0f} FPS", transform=ax3.transAxes,
                 color=QUANTUM_COLOR, fontsize=8)

        ax3.set_xlabel("Environment Steps")
        ax3.set_ylabel("Frames Per Second (FPS)")
        ax3.set_title("Effective Throughput (FPS)\n(higher = faster training)")
        ax3.legend()
        ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    else:
        _placeholder_panel(ax3, "FPS Comparison\n(No timing data in logs)")

    # ── Panel 4: Time per 1000 steps bar chart ───────────────────────────
    ax4 = axes[1, 1]

    if has_timing:
        # Compute average time per update
        q_avg_rollout = np.mean(q_rollout["values"])
        c_avg_rollout = np.mean(c_rollout["values"])
        q_avg_update  = np.mean(q_update["values"])
        c_avg_update  = np.mean(c_update["values"])

        categories = ["Rollout\n(env interaction)", "Gradient Update\n(VQC vs NN backprop)"]
        q_times = [q_avg_rollout, q_avg_update]
        c_times = [c_avg_rollout, c_avg_update]

        x = np.arange(len(categories))
        w = 0.35
        bars_q = ax4.bar(x - w/2, q_times, w, color=QUANTUM_COLOR,
                         alpha=0.85, label="Quantum PPO", zorder=3)
        bars_c = ax4.bar(x + w/2, c_times, w, color=CLASSICAL_COLOR,
                         alpha=0.85, label="Classical PPO", zorder=3)

        # Value labels on bars
        for bar in bars_q + bars_c:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(q_times + c_times) * 0.01,
                f"{bar.get_height():.2f}s",
                ha="center", va="bottom", fontsize=9,
            )

        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel("Average Time (seconds)")
        ax4.set_title(
            "Average Time per Update Phase\n(lower = computationally cheaper)"
        )
        ax4.legend()

        # Speedup annotation
        if q_avg_update > 0:
            speedup = q_avg_update / max(c_avg_update, 1e-8)
            ax4.text(
                0.97, 0.97,
                f"VQC is {speedup:.1f}× {'slower' if speedup > 1 else 'faster'}\nthan classical backprop",
                transform=ax4.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
            )
    else:
        _placeholder_panel(ax4, "Time per Update\n(No timing data in logs)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _placeholder_panel(ax, message: str) -> None:
    """Draw a placeholder panel when data is unavailable."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", fontsize=12,
            color="gray",
            bbox=dict(boxstyle="round", facecolor="#F0F0F0"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: FULL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def plot_full_dashboard(
    q_steps, q_rewards,
    c_steps, c_rewards,
    q_data: dict,
    c_data: dict,
    output_path: str,
    window: int = 50,
) -> None:
    """
    Figure 4: Comprehensive 4-panel research dashboard.

    Panel 1 (top-left):  Reward convergence with ±σ band
    Panel 2 (top-right): Stability — rolling std over training
    Panel 3 (bottom-left): Computational cost bar chart
    Panel 4 (bottom-right): Final performance summary box
    """
    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    fig.suptitle(
        "Quantum PPO vs Classical PPO — Research Dashboard\n"
        "CartPole-v1  |  VQC (Data Re-uploading)  |  PPO-Clip",
        fontsize=16, fontweight="bold",
    )

    # ── Panel 1: Reward Convergence ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    q_mean, q_std_r = rolling_stats(q_rewards, window)
    c_mean, c_std_r = rolling_stats(c_rewards, window)

    ax1.fill_between(q_steps, q_mean - q_std_r, q_mean + q_std_r,
                     color=QUANTUM_COLOR, alpha=0.2)
    ax1.fill_between(c_steps, c_mean - c_std_r, c_mean + c_std_r,
                     color=CLASSICAL_COLOR, alpha=0.2)
    ax1.plot(q_steps, q_mean, color=QUANTUM_COLOR, lw=2.5,
             label=f"Quantum PPO")
    ax1.plot(c_steps, c_mean, color=CLASSICAL_COLOR, lw=2.5,
             label=f"Classical PPO")
    ax1.axhline(500, color=SOLVE_COLOR, ls=":", lw=1.5, alpha=0.8,
                label="Perfect (500)")

    ax1.set_title("① Reward Convergence\n(with ±1σ stability band)")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 560)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ── Panel 2: Rolling Std (Stability) ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    q_std_roll = rolling_std(q_rewards, window)
    c_std_roll = rolling_std(c_rewards, window)

    ax2.plot(q_steps, q_std_roll, color=QUANTUM_COLOR, lw=2,
             label="Quantum PPO — σ")
    ax2.plot(c_steps, c_std_roll, color=CLASSICAL_COLOR, lw=2,
             label="Classical PPO — σ")
    ax2.fill_between(q_steps, 0, q_std_roll, color=QUANTUM_COLOR, alpha=0.15)
    ax2.fill_between(c_steps, 0, c_std_roll, color=CLASSICAL_COLOR, alpha=0.15)

    ax2.set_title("② Policy Stability\n(rolling σ — narrowing = convergence)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Rolling Std σ")
    ax2.legend()
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ── Panel 3: Computational Cost ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    q_update = q_data.get("timing/update_sec", {})
    c_update = c_data.get("timing/update_sec", {})
    q_rollout = q_data.get("timing/rollout_sec", {})
    c_rollout = c_data.get("timing/rollout_sec", {})

    has_timing = all(
        len(d.get("values", [])) > 0
        for d in [q_rollout, q_update, c_rollout, c_update]
    )

    if has_timing:
        categories = ["Rollout", "Gradient\nUpdate", "Total\nper cycle"]
        q_avg_r = np.mean(q_rollout["values"])
        c_avg_r = np.mean(c_rollout["values"])
        q_avg_u = np.mean(q_update["values"])
        c_avg_u = np.mean(c_update["values"])
        q_times = [q_avg_r, q_avg_u, q_avg_r + q_avg_u]
        c_times = [c_avg_r, c_avg_u, c_avg_r + c_avg_u]

        x = np.arange(len(categories))
        w = 0.35
        ax3.bar(x - w/2, q_times, w, color=QUANTUM_COLOR, alpha=0.85,
                label="Quantum PPO", zorder=3)
        ax3.bar(x + w/2, c_times, w, color=CLASSICAL_COLOR, alpha=0.85,
                label="Classical PPO", zorder=3)
        for xi, (qt, ct) in enumerate(zip(q_times, c_times)):
            ax3.text(xi - w/2, qt, f"{qt:.2f}s", ha="center", va="bottom", fontsize=8)
            ax3.text(xi + w/2, ct, f"{ct:.2f}s", ha="center", va="bottom", fontsize=8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.set_ylabel("Time (seconds)")
        ax3.legend()
    else:
        _placeholder_panel(ax3, "③ Computational Cost\n(No timing data)")

    ax3.set_title("③ Computational Cost\n(time per training cycle)")

    # ── Panel 4: Summary Statistics Table ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    final_q_rewards = q_rewards[max(0, len(q_rewards) - 100):]
    final_c_rewards = c_rewards[max(0, len(c_rewards) - 100):]

    q_solve_idx = np.where(q_mean >= 450)[0]
    c_solve_idx = np.where(c_mean >= 450)[0]
    q_solve_step = f"{q_steps[q_solve_idx[0]]/1000:.0f}k" if len(q_solve_idx) > 0 else "N/A"
    c_solve_step = f"{c_steps[c_solve_idx[0]]/1000:.0f}k" if len(c_solve_idx) > 0 else "N/A"

    q_perfect = sum(1 for r in q_rewards if r >= 500)
    c_perfect = sum(1 for r in c_rewards if r >= 500)

    rows = [
        ["Metric",                    "Quantum PPO",                         "Classical PPO"],
        ["Total Episodes",            str(len(q_rewards)),                   str(len(c_rewards))],
        ["Best Reward",               f"{max(q_rewards):.0f}",               f"{max(c_rewards):.0f}"],
        ["Final μ (last 100 eps)",    f"{np.mean(final_q_rewards):.1f}",     f"{np.mean(final_c_rewards):.1f}"],
        ["Final σ (last 100 eps)",    f"{np.std(final_q_rewards):.1f}",      f"{np.std(final_c_rewards):.1f}"],
        ["Final σ² (last 100 eps)",   f"{np.var(final_q_rewards):.1f}",      f"{np.var(final_c_rewards):.1f}"],
        ["Stable solve step (≥450)",  q_solve_step,                          c_solve_step],
        ["Perfect episodes (=500)",   f"{q_perfect} ({100*q_perfect/len(q_rewards):.1f}%)",
                                      f"{c_perfect} ({100*c_perfect/len(c_rewards):.1f}%)"],
    ]

    col_widths = [0.45, 0.27, 0.27]
    col_starts = [0.01, 0.47, 0.73]
    row_h = 0.1
    header_h = 0.88

    for ri, row in enumerate(rows):
        y = header_h - ri * row_h
        for ci, (cell, start, width) in enumerate(zip(row, col_starts, col_widths)):
            weight = "bold" if ri == 0 else "normal"
            color = "white" if ri == 0 else ("white" if ri % 2 == 0 else "#F0F0F0")
            bg = "#333333" if ri == 0 else (QUANTUM_COLOR if (ri % 2 == 0) else CLASSICAL_COLOR
                  if ci > 0 else "white")

            if ri == 0:
                facecolor = "#2D3436"
            elif ci == 0:
                facecolor = "#ECEFF1" if ri % 2 == 0 else "#FAFAFA"
                color = "black"
            elif ci == 1:
                facecolor = f"{QUANTUM_COLOR}22" if ri % 2 == 0 else f"{QUANTUM_COLOR}11"
                color = "#3D3D3D"
            else:
                facecolor = f"{CLASSICAL_COLOR}22" if ri % 2 == 0 else f"{CLASSICAL_COLOR}11"
                color = "#3D3D3D"

            rect = plt.Rectangle(
                (start, y - row_h * 0.9), width, row_h * 0.88,
                transform=ax4.transAxes, facecolor=facecolor,
                edgecolor="white", linewidth=0.5, clip_on=False,
            )
            ax4.add_patch(rect)
            ax4.text(
                start + width / 2, y - row_h * 0.45,
                cell, transform=ax4.transAxes,
                ha="center", va="center", fontsize=8.5,
                color=color if ri == 0 else "black",
                fontweight=weight,
            )

    ax4.set_title("④ Performance Summary Table")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARK REPORT JSON
# ════════════════════════════════════════════════════════════════════════════

def generate_report(
    q_steps, q_rewards,
    c_steps, c_rewards,
    q_data: dict,
    c_data: dict,
    window: int = 50,
) -> dict:
    """Generate a machine-readable benchmark report."""
    q_mean, _ = rolling_stats(q_rewards, window)
    c_mean, _ = rolling_stats(c_rewards, window)

    # Solve steps (smoothed avg ≥ 450)
    q_solve_idx = np.where(q_mean >= 450)[0]
    c_solve_idx = np.where(c_mean >= 450)[0]

    # First perfect episode
    q_perf_idx = np.where(np.array(q_rewards) >= 500)[0]
    c_perf_idx = np.where(np.array(c_rewards) >= 500)[0]

    final_q = q_rewards[max(0, len(q_rewards) - 100):]
    final_c = c_rewards[max(0, len(c_rewards) - 100):]

    # Timing summary
    q_update_times = q_data.get("timing/update_sec", {}).get("values", [])
    c_update_times = c_data.get("timing/update_sec", {}).get("values", [])
    q_rollout_times = q_data.get("timing/rollout_sec", {}).get("values", [])
    c_rollout_times = c_data.get("timing/rollout_sec", {}).get("values", [])

    report = {
        "quantum": {
            "total_episodes": len(q_rewards),
            "total_steps": int(q_steps[-1]) if len(q_steps) > 0 else 0,
            "best_reward": float(max(q_rewards)),
            "final_mean_100": float(np.mean(final_q)),
            "final_std_100": float(np.std(final_q)),
            "final_var_100": float(np.var(final_q)),
            "stable_solve_step": int(q_steps[q_solve_idx[0]]) if len(q_solve_idx) > 0 else None,
            "first_perfect_step": int(q_steps[q_perf_idx[0]]) if len(q_perf_idx) > 0 else None,
            "first_perfect_episode": int(q_perf_idx[0] + 1) if len(q_perf_idx) > 0 else None,
            "perfect_episode_count": int(sum(1 for r in q_rewards if r >= 500)),
            "avg_update_time_sec": float(np.mean(q_update_times)) if len(q_update_times) > 0 else None,
            "avg_rollout_time_sec": float(np.mean(q_rollout_times)) if len(q_rollout_times) > 0 else None,
            "avg_fps": float(2048 / np.mean(q_rollout_times)) if len(q_rollout_times) > 0 else None,
        },
        "classical": {
            "total_episodes": len(c_rewards),
            "total_steps": int(c_steps[-1]) if len(c_steps) > 0 else 0,
            "best_reward": float(max(c_rewards)),
            "final_mean_100": float(np.mean(final_c)),
            "final_std_100": float(np.std(final_c)),
            "final_var_100": float(np.var(final_c)),
            "stable_solve_step": int(c_steps[c_solve_idx[0]]) if len(c_solve_idx) > 0 else None,
            "first_perfect_step": int(c_steps[c_perf_idx[0]]) if len(c_perf_idx) > 0 else None,
            "first_perfect_episode": int(c_perf_idx[0] + 1) if len(c_perf_idx) > 0 else None,
            "perfect_episode_count": int(sum(1 for r in c_rewards if r >= 500)),
            "avg_update_time_sec": float(np.mean(c_update_times)) if len(c_update_times) > 0 else None,
            "avg_rollout_time_sec": float(np.mean(c_rollout_times)) if len(c_rollout_times) > 0 else None,
            "avg_fps": float(2048 / np.mean(c_rollout_times)) if len(c_rollout_times) > 0 else None,
        },
    }

    # Comparative analysis
    q_solve = report["quantum"]["stable_solve_step"]
    c_solve = report["classical"]["stable_solve_step"]
    if q_solve and c_solve and q_solve > 0:
        report["comparison"] = {
            "sample_efficiency_ratio": c_solve / q_solve,
            "faster_convergence": "quantum" if q_solve < c_solve else "classical",
            "final_mean_winner": "quantum" if report["quantum"]["final_mean_100"] >
                                              report["classical"]["final_mean_100"] else "classical",
            "stability_winner": "quantum" if report["quantum"]["final_std_100"] <
                                             report["classical"]["final_std_100"] else "classical",
        }

        if report["quantum"]["avg_update_time_sec"] and report["classical"]["avg_update_time_sec"]:
            vqc_time = report["quantum"]["avg_update_time_sec"]
            nn_time = report["classical"]["avg_update_time_sec"]
            report["comparison"]["vqc_overhead_factor"] = vqc_time / max(nn_time, 1e-8)
    else:
        report["comparison"] = {}

    return report


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and analyze Quantum vs Classical PPO from TensorBoard logs",
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs",
        help="Directory containing TensorBoard run folders (default: runs)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="benchmark_data",
        help="Directory for output figures and CSVs (default: benchmark_data)"
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Smoothing window for rolling mean/std (default: 50)"
    )
    parser.add_argument(
        "--quantum_prefix", type=str, default="hybrid_ppo_",
        help="Prefix for quantum PPO run directories"
    )
    parser.add_argument(
        "--classical_prefix", type=str, default="classical_ppo_",
        help="Prefix for classical PPO run directories"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent.resolve()
    runs_dir = root / args.runs_dir
    out_dir = root / args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "═" * 60)
    print(" QUANTUM VS CLASSICAL PPO — BENCHMARKING")
    print("═" * 60)
    print(f"  Runs dir:  {runs_dir}")
    print(f"  Output:    {out_dir}")
    print(f"  Window:    {args.window} episodes")
    print("═" * 60)

    if not HAS_TENSORBOARD:
        print("[ERROR] TensorBoard not installed. Run: pip install tensorboard")
        return

    # ── Find runs ────────────────────────────────────────────────────────
    print("\n[1/6] Locating TensorBoard run directories...")
    try:
        q_dir = find_run(str(runs_dir), args.quantum_prefix)
        print(f"  Quantum:   {q_dir}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return

    try:
        c_dir = find_run(str(runs_dir), args.classical_prefix)
        print(f"  Classical: {c_dir}")
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return

    # ── Extract all scalars ───────────────────────────────────────────────
    print("\n[2/6] Extracting TensorBoard scalars...")
    q_data = extract_all_scalars(q_dir)
    c_data = extract_all_scalars(c_dir)
    print(f"  Quantum tags:   {sorted(q_data.keys())}")
    print(f"  Classical tags: {sorted(c_data.keys())}")

    # ── Export to CSV ─────────────────────────────────────────────────────
    print("\n[3/6] Exporting raw data to CSV...")
    export_to_csv(q_data, str(out_dir), "quantum")
    export_to_csv(c_data, str(out_dir), "classical")

    # ── Load episode rewards ──────────────────────────────────────────────
    q_rewards_data = q_data.get("episode/reward", {})
    c_rewards_data = c_data.get("episode/reward", {})

    if not q_rewards_data or not c_rewards_data:
        print("[ERROR] Could not find 'episode/reward' tag in one or both runs.")
        print("  Available quantum tags:", list(q_data.keys()))
        print("  Available classical tags:", list(c_data.keys()))
        return

    q_steps   = q_rewards_data["steps"]
    q_rewards = q_rewards_data["values"]
    c_steps   = c_rewards_data["steps"]
    c_rewards = c_rewards_data["values"]

    print(f"\n  Quantum:   {len(q_rewards)} episodes, "
          f"steps {q_steps[0]:.0f}–{q_steps[-1]:.0f}")
    print(f"  Classical: {len(c_rewards)} episodes, "
          f"steps {c_steps[0]:.0f}–{c_steps[-1]:.0f}")

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n[4/6] Generating figures...")

    plot_reward_convergence(
        q_steps, q_rewards,
        c_steps, c_rewards,
        output_path=str(out_dir / "fig1_reward_convergence.png"),
        window=args.window,
    )

    plot_stability_analysis(
        q_steps, q_rewards,
        c_steps, c_rewards,
        output_path=str(out_dir / "fig2_stability_analysis.png"),
        window=args.window,
    )

    plot_computational_cost(
        q_data, c_data,
        output_path=str(out_dir / "fig3_computational_cost.png"),
    )

    plot_full_dashboard(
        q_steps, q_rewards,
        c_steps, c_rewards,
        q_data, c_data,
        output_path=str(out_dir / "fig4_full_dashboard.png"),
        window=args.window,
    )

    # ── Generate benchmark report ─────────────────────────────────────────
    print("\n[5/6] Generating benchmark report JSON...")
    report = generate_report(q_steps, q_rewards, c_steps, c_rewards, q_data, c_data, args.window)
    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {report_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n[6/6] Summary")
    print("═" * 60)
    print(f"  {'Metric':<40} {'Quantum':>9} {'Classical':>9}")
    print("  " + "─" * 58)
    print(f"  {'Total Episodes':<40} {report['quantum']['total_episodes']:>9} "
          f"{report['classical']['total_episodes']:>9}")
    print(f"  {'Best Reward':<40} {report['quantum']['best_reward']:>9.0f} "
          f"{report['classical']['best_reward']:>9.0f}")
    print(f"  {'Final μ (last 100 episodes)':<40} {report['quantum']['final_mean_100']:>9.1f} "
          f"{report['classical']['final_mean_100']:>9.1f}")
    print(f"  {'Final σ (last 100 episodes)':<40} {report['quantum']['final_std_100']:>9.1f} "
          f"{report['classical']['final_std_100']:>9.1f}")
    print(f"  {'Final σ² (last 100 episodes)':<40} {report['quantum']['final_var_100']:>9.1f} "
          f"{report['classical']['final_var_100']:>9.1f}")

    q_sss = report['quantum']['stable_solve_step']
    c_sss = report['classical']['stable_solve_step']
    print(f"  {'Stable solve step (smoothed ≥450)':<40} "
          f"{f'{q_sss:,}' if q_sss else 'N/A':>9} "
          f"{f'{c_sss:,}' if c_sss else 'N/A':>9}")

    if "comparison" in report and report["comparison"]:
        comp = report["comparison"]
        print("\n  Comparative Analysis:")
        if "sample_efficiency_ratio" in comp:
            ratio = comp["sample_efficiency_ratio"]
            winner = comp["faster_convergence"]
            print(f"    Convergence speed ratio: {ratio:.2f}x  → {winner} converges faster")
        if "vqc_overhead_factor" in comp:
            ovhd = comp["vqc_overhead_factor"]
            print(f"    VQC gradient overhead:   {ovhd:.1f}x vs classical backprop")
        print(f"    Best final performance:  {comp.get('final_mean_winner', 'N/A')}")
        print(f"    Most stable (lower σ):   {comp.get('stability_winner', 'N/A')}")

    print("═" * 60)
    print(f"\n  Output files in: {out_dir}")
    for f in sorted(out_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
