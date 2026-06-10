"""
benchmark_qnpg.py — Ablation Study: Adam-PPO vs QNPG-PPO vs QNPG-SGD
======================================================================
Runs a systematic benchmark comparing:
  1. Baseline Adam-PPO       (qnpg_enabled=False)
  2. QNPG-PPO Diagonal       (qnpg_enabled=True, qfim_mode="diagonal")
  3. QNPG-PPO Block-Diagonal  (qnpg_enabled=True, qfim_mode="block_diag")
  4. QNPG-SGD (ablation)     (qnpg_enabled=True, use_natural_grad=False)

Usage:
    cd src/quantum
    python benchmark_qnpg.py --env HPCSchedulingEnv-v0 --timesteps 50000

Outputs:
    - TensorBoard logs in runs/benchmark_qnpg_<timestamp>/
    - Summary CSV: outputs/qnpg_benchmark_results.csv
    - Convergence curves comparing sample efficiency across methods

Results tracked:
    - Episode reward (main performance metric)
    - Convergence speed (steps to 95% max reward)
    - QFIM diagnostics (mean diagonal, condition number, effective dim)
    - Gradient norms (Euclidean vs Natural)
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import List, Dict

import numpy as np
import torch

# Allow direct execution from src/quantum
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from train import train


# ════════════════════════════════════════════════════════════════════════
# Experiment Configurations
# ════════════════════════════════════════════════════════════════════════

def make_base_config(
    env_name: str,
    total_timesteps: int,
    seed: int,
    n_qubits: int = 4,
    n_layers: int = 2,
) -> Config:
    """Create base configuration shared across all benchmark variants."""
    return Config(
        env_name=env_name,
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding_type="data_reuploading",
        total_timesteps=total_timesteps,
        rollout_steps=512,        # Shorter rollouts for faster benchmarking
        mini_batch_size=64,
        ppo_epochs=4,
        actor_lr=5e-3,
        critic_lr=1e-3,
        seed=seed,
        diagnose_barren_plateau=True,
        diagnose_interval=5,
        log_dir="runs/benchmark_qnpg",
        checkpoint_dir="checkpoints/benchmark_qnpg",
    )


VARIANTS = [
    # (label, qnpg_enabled, qfim_mode, use_natural_grad)
    ("Adam-PPO",           False, "diagonal",   True),
    ("QNPG-Diagonal",      True,  "diagonal",   True),
    ("QNPG-BlockDiag",     True,  "block_diag", True),
    ("QNPG-SGD-Ablation",  True,  "diagonal",   False),  # Ablation: QNPG struct, no preconditioning
]


# ════════════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ════════════════════════════════════════════════════════════════════════

def run_benchmark(
    env_name: str,
    total_timesteps: int,
    seeds: List[int],
    n_qubits: int,
    n_layers: int,
    output_dir: str = "outputs",
) -> None:
    """
    Run all benchmark variants across multiple seeds.

    Args:
        env_name: Gymnasium environment name.
        total_timesteps: Training budget per run.
        seeds: List of random seeds for statistical significance.
        n_qubits: Number of qubits in the VQC.
        n_layers: Number of VQC layers.
        output_dir: Directory to save benchmark results.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"qnpg_benchmark_{timestamp}.csv")

    print("=" * 70)
    print("  QNPG Benchmark Study: Adam-PPO vs QNPG-PPO")
    print("=" * 70)
    print(f"  Environment:    {env_name}")
    print(f"  Total Steps:    {total_timesteps:,}")
    print(f"  Seeds:          {seeds}")
    print(f"  Qubits:         {n_qubits}")
    print(f"  VQC Layers:     {n_layers}")
    print(f"  Variants:       {len(VARIANTS)}")
    print(f"  Total runs:     {len(VARIANTS) * len(seeds)}")
    print("=" * 70 + "\n")

    results = []

    for label, qnpg_enabled, qfim_mode, use_nat_grad in VARIANTS:
        for seed in seeds:
            print(f"\n{'─' * 60}")
            print(f"  Running: {label} | seed={seed}")
            print(f"{'─' * 60}")

            # Build config for this variant
            cfg = make_base_config(env_name, total_timesteps, seed, n_qubits, n_layers)
            cfg.qnpg_enabled = qnpg_enabled
            cfg.qnpg_qfim_mode = qfim_mode
            cfg.qnpg_use_natural_grad = use_nat_grad
            cfg.qnpg_n_samples = 4       # Diagonal: 4 samples is enough
            cfg.qnpg_damping = 1e-3      # Standard damping
            cfg.qnpg_block_size = n_qubits * 3  # One block per VQC layer

            run_start = time.time()
            try:
                train(cfg)
                run_time = time.time() - run_start
                status = "OK"
            except Exception as e:
                print(f"  ⚠️  Run failed: {e}")
                run_time = time.time() - run_start
                status = f"ERROR: {e}"

            results.append({
                "variant": label,
                "qnpg_enabled": qnpg_enabled,
                "qfim_mode": qfim_mode,
                "use_natural_grad": use_nat_grad,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "run_time_sec": round(run_time, 1),
                "status": status,
            })

            # Save intermediate results
            _save_results(results, results_path)

    print(f"\n✅ Benchmark complete! Results saved to: {results_path}")
    _print_summary(results)


def _save_results(results: List[Dict], path: str) -> None:
    """Save results to CSV."""
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results: List[Dict]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<25} {'Seeds':<8} {'Avg Time(s)':<14} {'Status':<10}")
    print("-" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["variant"]].append(r)

    for variant, runs in grouped.items():
        times = [r["run_time_sec"] for r in runs if r["status"] == "OK"]
        n_ok = sum(1 for r in runs if r["status"] == "OK")
        avg_time = np.mean(times) if times else 0
        print(f"  {variant:<23} {n_ok}/{len(runs):<6} {avg_time:<14.1f} {'✓' if n_ok == len(runs) else '⚠'}")

    print("=" * 70)
    print("\nTo compare results, run:")
    print("  tensorboard --logdir runs/benchmark_qnpg")


# ════════════════════════════════════════════════════════════════════════
# QFIM Analysis Script
# ════════════════════════════════════════════════════════════════════════

def analyze_qfim(
    env_name: str = "CartPole-v1",
    n_qubits: int = 4,
    n_layers: int = 2,
    n_states: int = 20,
) -> None:
    """
    Standalone QFIM analysis: estimate QFIM for a freshly initialized VQC.

    Prints:
    - Diagonal QFIM entries (parameter sensitivity)
    - Condition number (ill-conditioning indicator)
    - Effective dimension (active parameter count)

    Useful to understand the geometry of the quantum parameter space
    before training, and to choose the appropriate QFIM mode.

    Args:
        env_name: Environment to sample states from.
        n_qubits: Qubit count.
        n_layers: VQC layer count.
        n_states: Number of random states for QFIM estimation.
    """
    import gymnasium as gym
    from quantum_actor import QuantumActor
    from qfim import QFIMEstimator, QFIMMode

    print("=" * 60)
    print("  QFIM Analysis")
    print("=" * 60)

    # Build a minimal config and actor
    cfg = Config(
        env_name=env_name,
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding_type="data_reuploading",
    )
    env = gym.make(env_name)
    obs_space = env.observation_space

    # Auto-detect dims
    import numpy as np
    from gymnasium.spaces import Box, Discrete
    if isinstance(obs_space, Box):
        cfg.state_dim = int(np.prod(obs_space.shape))
    elif isinstance(obs_space, Discrete):
        cfg.state_dim = int(obs_space.n)
    cfg.action_dim = (
        int(env.action_space.n)
        if isinstance(env.action_space, Discrete)
        else int(np.prod(env.action_space.shape))
    )

    actor = QuantumActor(cfg, obs_space)
    estimator = QFIMEstimator(actor, mode=QFIMMode.DIAGONAL, n_samples=n_states)

    # Sample random states
    states = torch.tensor(
        np.array([obs_space.sample() for _ in range(n_states)]),
        dtype=torch.float32
    )

    print(f"\n  VQC: {n_qubits} qubits × {n_layers} layers")
    print(f"  Quantum params (d): {estimator._d}")
    print(f"  States sampled: {n_states}")
    print()

    stats = estimator.get_qfim_stats(states)
    diag = estimator.estimate_diagonal_qfim(states)

    print(f"  Mean QFIM diagonal: {stats['mean_diag']:.6f}")
    print(f"  Max QFIM diagonal:  {stats['max_diag']:.6f}")
    print(f"  Min QFIM diagonal:  {stats['min_diag']:.6f}")
    print(f"  Condition number κ: {stats['condition_number']:.2f}")
    print(f"  Effective dim:      {stats['effective_dim']}/{estimator._d}")
    print()
    print("  Interpretation:")
    if stats['mean_diag'] < 1e-3:
        print("  ⚠️  Very low QFIM diagonals → possible barren plateau region")
    else:
        print("  ✓  QFIM diagonals are non-trivial → gradient landscape is active")

    if stats['condition_number'] > 100:
        print("  ⚠️  High condition number → consider increasing damping")
    else:
        print("  ✓  Condition number is manageable")

    env.close()


# ════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QNPG Benchmark: Adam-PPO vs QNPG-PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env", default="CartPole-v1",
        help="Gymnasium environment name"
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000,
        help="Total environment steps per run"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds for multiple runs"
    )
    parser.add_argument(
        "--n-qubits", type=int, default=4,
        help="Number of qubits in VQC"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help="Number of VQC variational layers"
    )
    parser.add_argument(
        "--analyze-qfim", action="store_true",
        help="Only run QFIM analysis (no training)"
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Directory for benchmark output CSV"
    )

    args = parser.parse_args()

    if args.analyze_qfim:
        analyze_qfim(args.env, args.n_qubits, args.n_layers)
    else:
        run_benchmark(
            env_name=args.env,
            total_timesteps=args.timesteps,
            seeds=args.seeds,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            output_dir=args.output_dir,
        )
