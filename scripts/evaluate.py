"""
evaluate.py — Inference & Evaluation Script
============================================
Loads trained model weights (no gradient computation) and evaluates both the
Quantum PPO agent and the Classical PPO agent on CartPole-v1.

Features:
  1. Deterministic action selection (argmax of policy distribution)
  2. Video recording via gym.wrappers.RecordVideo (.mp4 output)
  3. 100-episode statistical evaluation (mean μ, variance σ², std σ)
  4. Side-by-side comparison of quantum vs classical performance
  5. Saves all results to JSON for downstream analysis
  6. Backward-compatible loader for old checkpoint format (pre-PreEncodingNN)

Usage:
    # Evaluate both agents (default: 100 episodes)
    python evaluate.py

    # Evaluate quantum agent only
    python evaluate.py --agent quantum --n_episodes 100

    # Evaluate classical agent only
    python evaluate.py --agent classical --n_episodes 50

    # With video recording
    python evaluate.py --video_dir eval_videos

    # Custom checkpoint paths
    python evaluate.py \\
        --quantum_ckpt checkpoints/hybrid_ppo_final.pt \\
        --classical_ckpt checkpoints/classical_ppo_final.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ── Path setup: allow importing from sub-packages ───────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
QUANTUM_MODEL_DIR = ROOT_DIR / "quantum_model"
CLASSICAL_MODEL_DIR = ROOT_DIR / "classical_model"


# ════════════════════════════════════════════════════════════════════════════
# LEGACY QUANTUM ACTOR (for backward-compatible checkpoint loading)
# ════════════════════════════════════════════════════════════════════════════

def _build_legacy_quantum_actor(n_qubits: int, n_layers: int, action_dim: int):
    """
    Reconstruct the original simple QuantumActor (pre-PreEncodingNN era).

    The checkpoint from hybrid_ppo_final.pt was saved from a simpler architecture:
      - Raw 4-dim CartPole state fed directly into the quantum circuit
      - No classical pre-encoding network
      - Data re-uploading encoding (RY gates)
      - output_map (not post_nn) for the final Linear layer
      - input_scaling as nn.Parameter (always trainable)

    This matches keys: ['q_params', 'input_scaling', 'output_map.weight', 'output_map.bias']
    """
    import pennylane as qml
    from torch.distributions import Categorical

    class LegacyQuantumActor(torch.nn.Module):
        """
        Original VQC actor: raw state → data re-uploading circuit → output_map.

        This exactly mirrors the architecture used when hybrid_ppo_final.pt was saved.
        """

        def __init__(self):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.action_dim = action_dim

            # ── PennyLane device ─────────────────────────────────────────
            try:
                self.qdev = qml.device("lightning.gpu", wires=n_qubits)
                print("    Using lightning.gpu device.")
            except Exception:
                try:
                    self.qdev = qml.device("lightning.qubit", wires=n_qubits)
                    print("    Using lightning.qubit device.")
                except Exception:
                    self.qdev = qml.device("default.qubit", wires=n_qubits)
                    print("    Using default.qubit device.")

            # ── Quantum parameters ───────────────────────────────────────
            self.q_params = torch.nn.Parameter(
                torch.zeros(n_layers, n_qubits, 3, dtype=torch.float32)
            )
            self.input_scaling = torch.nn.Parameter(
                torch.ones(n_layers, n_qubits, dtype=torch.float32)
            )

            # ── Classical output layer (matches checkpoint key 'output_map') ─
            self.output_map = torch.nn.Linear(n_qubits, action_dim)

            # ── Build QNode ──────────────────────────────────────────────
            def circuit(inputs, weights, scaling):
                for layer in range(weights.shape[0]):
                    # Data re-uploading: encode state at each layer
                    for qubit in range(n_qubits):
                        qml.RY(scaling[layer, qubit] * inputs[qubit], wires=qubit)
                    # Trainable rotations
                    for qubit in range(n_qubits):
                        qml.RX(weights[layer, qubit, 0], wires=qubit)
                        qml.RY(weights[layer, qubit, 1], wires=qubit)
                        qml.RZ(weights[layer, qubit, 2], wires=qubit)
                    # CNOT entanglement chain
                    for qubit in range(n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.qnode = qml.QNode(
                circuit, self.qdev,
                interface="torch",
                diff_method="adjoint",
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            if state.dim() == 1:
                measurements = self.qnode(state, self.q_params, self.input_scaling)
                meas_tensor = torch.stack(list(measurements)).float()
                return self.output_map(meas_tensor)
            else:
                results = []
                for i in range(state.shape[0]):
                    m = self.qnode(state[i], self.q_params, self.input_scaling)
                    results.append(self.output_map(torch.stack(list(m)).float()))
                return torch.stack(results)

        def get_distribution(self, state: torch.Tensor):
            return Categorical(logits=self.forward(state))

    return LegacyQuantumActor()


# ════════════════════════════════════════════════════════════════════════════
# QUANTUM AGENT LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_quantum_actor(checkpoint_path: str, device: torch.device):
    """
    Load the QuantumActor from a saved checkpoint.

    Automatically detects old vs new checkpoint format:
    - OLD: keys = ['q_params', 'input_scaling', 'output_map.weight/bias']
    - NEW: keys include 'pre_encoding_nn.*', 'post_nn.*'

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device to load onto.

    Returns:
        Tuple of (actor, config) with actor in eval mode.
    """
    sys.path.insert(0, str(QUANTUM_MODEL_DIR))

    print(f"  Loading quantum checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    actor_sd = checkpoint["actor_state_dict"]

    print(f"  Config — env: {config.env_name}, qubits: {config.n_qubits}, "
          f"layers: {config.n_layers}")
    print(f"  Checkpoint keys: {list(actor_sd.keys())}")

    # ── Format detection ─────────────────────────────────────────────────
    is_old_format = (
        "output_map.weight" in actor_sd
        and "pre_encoding_nn.net.0.weight" not in actor_sd
    )

    if is_old_format:
        print("  → Detected OLD checkpoint format (no PreEncodingNN)")
        actor = _build_legacy_quantum_actor(
            config.n_qubits, config.n_layers, config.action_dim
        )
        missing, unexpected = actor.load_state_dict(actor_sd, strict=False)
        if missing:
            print(f"  [INFO] Missing keys (kept at init values): {missing}")
        if unexpected:
            print(f"  [WARN] Unexpected keys (ignored): {unexpected}")
    else:
        # New format with PreEncodingNN
        from quantum_actor import QuantumActor
        print(f"  → Detected NEW checkpoint format (encoding: {getattr(config, 'encoding_type', 'data_reuploading')})")
        env_tmp = gym.make(config.env_name)
        obs_space = env_tmp.observation_space
        env_tmp.close()
        actor = QuantumActor(config, obs_space)
        actor.load_state_dict(actor_sd)

    actor.eval()
    actor.to(device)
    print(f"  Quantum actor loaded — {sum(p.numel() for p in actor.parameters())} params")
    return actor, config


# ════════════════════════════════════════════════════════════════════════════
# CLASSICAL AGENT LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_classical_actor(checkpoint_path: str, device: torch.device):
    """
    Load the ClassicalActor from a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device to load onto.

    Returns:
        Tuple of (actor, config) with actor in eval mode.
    """
    sys.path.insert(0, str(CLASSICAL_MODEL_DIR))
    from classical_actor import ClassicalActor

    print(f"  Loading classical checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    print(f"  Config — env: {config.env_name}, state_dim: {config.state_dim}, "
          f"action_dim: {config.action_dim}")

    actor = ClassicalActor(config)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    actor.to(device)

    print(f"  Classical actor loaded — {sum(p.numel() for p in actor.parameters())} params")
    return actor, config


# ════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC ACTION SELECTION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def select_action_deterministic(actor, state: np.ndarray) -> int:
    """
    Choose the action deterministically (greedy / argmax) — no sampling.

    During inference, we select the most probable action rather than
    sampling from the distribution. This yields the best achievable
    policy performance and removes stochastic variability.

    For discrete actions: argmax over logits (equivalent to argmax over probabilities)

    Args:
        actor: The actor network (any class with a forward() method).
        state: Raw observation from environment.

    Returns:
        Deterministic integer action.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # forward() gives raw logits for discrete actions
    # argmax on logits == argmax on softmax(logits) since softmax is monotone
    output = actor(state_tensor)
    action = output.argmax(dim=-1).item()
    return action


# ════════════════════════════════════════════════════════════════════════════
# SINGLE EPISODE RUNNER
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_episode(
    actor,
    env: gym.Env,
) -> Tuple[float, int]:
    """
    Run a single deterministic episode.

    Args:
        actor: Trained actor network.
        env: Gymnasium environment (may be wrapped with RecordVideo).

    Returns:
        Tuple of (total_reward, episode_length).
    """
    state, _ = env.reset()
    total_reward = 0.0
    episode_length = 0
    done = False

    while not done:
        action = select_action_deterministic(actor, state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_length += 1
        done = terminated or truncated

    return total_reward, episode_length


# ════════════════════════════════════════════════════════════════════════════
# MULTI-EPISODE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_agent(
    actor,
    env_name: str,
    n_episodes: int,
    video_dir: Optional[str] = None,
    seed: int = 42,
    agent_label: str = "Agent",
) -> dict:
    """
    Evaluate an agent over n_episodes and collect statistics.

    Records a video of the first 3 episodes if video_dir is set.

    Args:
        actor: Trained actor network (in eval mode).
        env_name: Gymnasium environment name.
        n_episodes: Number of episodes to run.
        video_dir: Directory to save .mp4 video. None to skip.
        seed: Random seed for environment.
        agent_label: Label for progress display.

    Returns:
        Dictionary with evaluation statistics:
            - rewards: list of per-episode rewards
            - lengths: list of per-episode lengths
            - mean_reward: μ
            - std_reward: σ
            - var_reward: σ²
            - min_reward: minimum episode reward
            - max_reward: maximum episode reward
            - mean_length: average episode length
            - video_path: path to video (if recorded)
    """
    rewards = []
    lengths = []
    video_path = None

    print(f"\n{'─' * 55}")
    print(f"  Evaluating {agent_label} — {n_episodes} episodes")
    print(f"{'─' * 55}")

    # ── Video recording: wrap env for first 3 episodes ────────────────
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        video_env = gym.make(env_name, render_mode="rgb_array")
        video_env = RecordVideo(
            video_env,
            video_folder=video_dir,
            episode_trigger=lambda ep: ep < 3,   # Record episodes 0, 1, 2
            name_prefix=agent_label.lower().replace(" ", "_"),
            disable_logger=True,
        )
        video_env.reset(seed=seed)

        print(f"  Recording video to: {video_dir}/")
        n_video = min(3, n_episodes)
        for ep in range(n_video):
            r, l = run_episode(actor, video_env)
            rewards.append(r)
            lengths.append(l)
            print(f"    [VIDEO] Episode {ep+1:3d}: reward={r:6.1f}, length={l}")

        video_env.close()

        # Find generated video file
        mp4_files = list(Path(video_dir).glob("*.mp4"))
        if mp4_files:
            video_path = str(sorted(mp4_files)[-1])
            print(f"  Video saved: {video_path}")

        start_ep = len(rewards)
    else:
        start_ep = 0

    # ── Remaining episodes (non-video) ──────────────────────────────────
    plain_env = gym.make(env_name)
    plain_env.reset(seed=seed + 1000)

    for ep in range(start_ep, n_episodes):
        r, l = run_episode(actor, plain_env)
        rewards.append(r)
        lengths.append(l)

        # Progress display every 10 episodes
        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            current_mean = np.mean(rewards)
            current_std = np.std(rewards)
            print(
                f"    Episode {ep+1:4d}/{n_episodes} | "
                f"Reward: {r:6.1f} | "
                f"Running μ={current_mean:6.1f} ± {current_std:5.1f}"
            )

    plain_env.close()

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    stats = {
        "rewards": rewards.tolist(),
        "lengths": lengths.tolist(),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "var_reward": float(np.var(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "video_path": video_path,
        "n_episodes": n_episodes,
        "agent_label": agent_label,
    }

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n  {'─' * 45}")
    print(f"  RESULTS — {agent_label}")
    print(f"  {'─' * 45}")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Mean reward μ: {stats['mean_reward']:.2f}")
    print(f"  Std  reward σ: {stats['std_reward']:.2f}")
    print(f"  Var  reward σ²:{stats['var_reward']:.2f}")
    print(f"  Min reward:    {stats['min_reward']:.1f}")
    print(f"  Max reward:    {stats['max_reward']:.1f}")
    print(f"  Mean length:   {stats['mean_length']:.1f} steps")
    perfect = int(np.sum(rewards >= 500))
    print(f"  Perfect (≥500):{perfect}/{n_episodes} ({100*perfect/n_episodes:.1f}%)")

    return stats


# ════════════════════════════════════════════════════════════════════════════
# RESULTS SAVING
# ════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Quantum and/or Classical PPO agents on CartPole-v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent", choices=["quantum", "classical", "both"], default="both",
        help="Which agent(s) to evaluate (default: both)"
    )
    parser.add_argument(
        "--quantum_ckpt", type=str,
        default="checkpoints/hybrid_ppo_final.pt",
        help="Path to quantum PPO checkpoint"
    )
    parser.add_argument(
        "--classical_ckpt", type=str,
        default="checkpoints/classical_ppo_final.pt",
        help="Path to classical PPO checkpoint"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=100,
        help="Number of episodes to evaluate (default: 100)"
    )
    parser.add_argument(
        "--video_dir", type=str, default="eval_videos",
        help="Directory to save evaluation videos (default: eval_videos)"
    )
    parser.add_argument(
        "--no_video", action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json",
        help="Path to save JSON results (default: eval_results.json)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: cpu or cuda (default: cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("\n" + "═" * 60)
    print(" QUANTUM VS CLASSICAL PPO — INFERENCE EVALUATION")
    print("═" * 60)
    print(f"  Mode:       {args.agent}")
    print(f"  Episodes:   {args.n_episodes}")
    print(f"  Device:     {device}")
    print(f"  Seed:       {args.seed}")
    print(f"  Video:      {'disabled' if args.no_video else args.video_dir}")
    print("═" * 60)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_episodes": args.n_episodes,
        "seed": args.seed,
    }

    start_total = time.time()

    # ── Evaluate Quantum Agent ───────────────────────────────────────────
    if args.agent in ("quantum", "both"):
        q_ckpt = ROOT_DIR / args.quantum_ckpt
        if not q_ckpt.exists():
            print(f"\n  [WARNING] Quantum checkpoint not found: {q_ckpt}")
            print("  Skipping quantum evaluation.")
        else:
            t0 = time.time()
            quantum_actor, q_config = load_quantum_actor(str(q_ckpt), device)
            load_time = time.time() - t0
            print(f"  Load time: {load_time:.1f}s")

            video_dir = None if args.no_video else str(ROOT_DIR / args.video_dir / "quantum")
            q_stats = evaluate_agent(
                actor=quantum_actor,
                env_name=q_config.env_name,
                n_episodes=args.n_episodes,
                video_dir=video_dir,
                seed=args.seed,
                agent_label="Quantum PPO",
            )
            q_stats["checkpoint"] = str(q_ckpt)
            q_stats["load_time_sec"] = load_time
            all_results["quantum"] = q_stats

    # ── Evaluate Classical Agent ─────────────────────────────────────────
    if args.agent in ("classical", "both"):
        c_ckpt = ROOT_DIR / args.classical_ckpt
        if not c_ckpt.exists():
            print(f"\n  [WARNING] Classical checkpoint not found: {c_ckpt}")
            print("  Skipping classical evaluation.")
        else:
            t0 = time.time()
            classical_actor, c_config = load_classical_actor(str(c_ckpt), device)
            load_time = time.time() - t0
            print(f"  Load time: {load_time:.1f}s")

            video_dir = None if args.no_video else str(ROOT_DIR / args.video_dir / "classical")
            c_stats = evaluate_agent(
                actor=classical_actor,
                env_name=c_config.env_name,
                n_episodes=args.n_episodes,
                video_dir=video_dir,
                seed=args.seed,
                agent_label="Classical PPO",
            )
            c_stats["checkpoint"] = str(c_ckpt)
            c_stats["load_time_sec"] = load_time
            all_results["classical"] = c_stats

    total_time = time.time() - start_total

    # ── Side-by-side Comparison ──────────────────────────────────────────
    if "quantum" in all_results and "classical" in all_results:
        q = all_results["quantum"]
        c = all_results["classical"]

        print("\n" + "═" * 65)
        print(" SIDE-BY-SIDE COMPARISON")
        print("═" * 65)
        print(f"  {'Metric':<35} {'Quantum PPO':>14} {'Classical PPO':>14}")
        print("  " + "─" * 63)
        print(f"  {'Mean Reward μ':<35} {q['mean_reward']:>14.2f} {c['mean_reward']:>14.2f}")
        print(f"  {'Std Reward σ':<35} {q['std_reward']:>14.2f} {c['std_reward']:>14.2f}")
        print(f"  {'Variance σ²':<35} {q['var_reward']:>14.2f} {c['var_reward']:>14.2f}")
        print(f"  {'Min Reward':<35} {q['min_reward']:>14.1f} {c['min_reward']:>14.1f}")
        print(f"  {'Max Reward':<35} {q['max_reward']:>14.1f} {c['max_reward']:>14.1f}")
        print(f"  {'Mean Ep Length':<35} {q['mean_length']:>14.1f} {c['mean_length']:>14.1f}")

        q_perfect = sum(1 for r in q["rewards"] if r >= 500)
        c_perfect = sum(1 for r in c["rewards"] if r >= 500)
        print(f"  {'Perfect Episodes (≥500)':<35} {q_perfect:>14} {c_perfect:>14}")
        print(f"  {'Perfect Rate':<35} {100*q_perfect/q['n_episodes']:>13.1f}% "
              f"{100*c_perfect/c['n_episodes']:>13.1f}%")
        print("═" * 65)

        # Stability analysis
        q_cv = q["std_reward"] / max(q["mean_reward"], 1e-8) * 100
        c_cv = c["std_reward"] / max(c["mean_reward"], 1e-8) * 100
        print(f"\n  Coefficient of Variation (lower = more stable):")
        print(f"    Quantum PPO:   {q_cv:.1f}%")
        print(f"    Classical PPO: {c_cv:.1f}%")

        winner = "Quantum PPO" if q["mean_reward"] > c["mean_reward"] else "Classical PPO"
        print(f"\n  Winner by mean reward: {winner}")

    print(f"\n  Total evaluation time: {total_time:.1f}s")

    # ── Save JSON results ─────────────────────────────────────────────────
    all_results["total_eval_time_sec"] = total_time
    output_path = ROOT_DIR / args.output
    save_results(all_results, str(output_path))

    print("\n" + "═" * 60)
    print(f"  Evaluation complete!")
    if not args.no_video:
        print(f"  Videos: {ROOT_DIR / args.video_dir}/")
    print(f"  Results JSON: {output_path}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
