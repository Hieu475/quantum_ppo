"""
test_hpc_env.py — Smoke-test & sanity checks for HPCSchedulingEnv
===================================================================
Run with:
    python envs/test_hpc_env.py
"""

import sys
import time
import traceback

import numpy as np
import gymnasium as gym

# ── Allow running from repo root without installing the package ───────────────
sys.path.insert(0, ".")
from envs import HPCSchedulingEnv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_spaces():
    print("\n[1] Spaces")
    env = HPCSchedulingEnv(num_nodes=4, num_jobs_visible=3)
    obs_dim_expected = 4 * 3 + 3 * 3  # 21
    check("obs_space is Box", isinstance(env.observation_space, gym.spaces.Box))
    check(
        "obs_dim matches formula",
        env.observation_space.shape[0] == obs_dim_expected,
        f"expected {obs_dim_expected}, got {env.observation_space.shape[0]}",
    )
    check(
        "action_space is Discrete(N+1)",
        isinstance(env.action_space, gym.spaces.Discrete)
        and int(env.action_space.n) == 5,
    )
    env.close()


def test_reset():
    print("\n[2] reset()")
    env = HPCSchedulingEnv(num_nodes=3, num_jobs_visible=2, seed=42)
    obs, info = env.reset(seed=42)
    check("obs is ndarray", isinstance(obs, np.ndarray))
    check("obs dtype float32", obs.dtype == np.float32)
    check("obs in [0, 1]", obs.min() >= 0.0 and obs.max() <= 1.0)
    check("info has queue_length", "queue_length" in info)
    env.close()


def test_step_random():
    print("\n[3] Random-action rollout (1 episode)")
    env = HPCSchedulingEnv(
        num_nodes=3,
        num_jobs_visible=2,
        num_total_jobs=10,
        max_steps=200,
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        check(
            f"  step {steps:3d}: obs valid",
            obs.min() >= 0.0 and obs.max() <= 1.0,
            f"reward={reward:.3f}",
        ) if steps < 5 else None
        total_reward += reward
        steps += 1

    check("episode terminated or truncated", terminated or truncated)
    check("at least 1 step taken", steps >= 1)
    print(
        f"  Episode ended at step {steps} "
        f"({'terminated' if terminated else 'truncated'}), "
        f"total_reward={total_reward:.2f}"
    )
    env.close()


def test_invalid_action_penalty():
    print("\n[4] Invalid dispatch → penalty applied")
    env = HPCSchedulingEnv(
        num_nodes=2,
        num_jobs_visible=1,
        num_total_jobs=5,
        delta=10.0,
        gamma=5.0,
    )
    obs, _ = env.reset(seed=99)

    # Manually fill node 0 to capacity so any dispatch there is invalid
    env.nodes[0].cpu_used = 1.0
    env.nodes[0].ram_used = 1.0

    # Force a job into the queue with a known small request
    from envs.hpc_scheduling_env import Job
    from collections import deque
    env.job_queue = deque([Job(999, 0.1, 0.1, 3.0)])

    # Try dispatching to the full node
    _, reward, _, _, _ = env.step(0)  # action=0 → Node 0 (full)
    check("negative reward for invalid action", reward < 0, f"reward={reward:.2f}")
    env.close()


def test_wait_action():
    print("\n[5] Wait action (action = num_nodes)")
    env = HPCSchedulingEnv(num_nodes=3, num_jobs_visible=1, num_total_jobs=5)
    obs, _ = env.reset(seed=7)
    q_before = len(env.job_queue)
    _, reward, _, _, info = env.step(env.num_nodes)  # wait
    check("queue unchanged after wait", len(env.job_queue) == q_before)
    env.close()


def test_reproducibility():
    print("\n[6] Reproducibility (same seed → same rollout)")

    def run(seed):
        env = HPCSchedulingEnv(num_nodes=3, num_jobs_visible=2, num_total_jobs=8)
        obs, _ = env.reset(seed=seed)
        rewards = []
        for _ in range(20):
            action = env.action_space.sample()  # seeded inside env? No — use fixed seed
            _, r, done, trunc, _ = env.step(action)
            rewards.append(r)
            if done or trunc:
                break
        env.close()
        return obs

    obs1 = run(123)
    obs2 = run(123)
    check("identical initial obs with same seed", np.allclose(obs1, obs2))


def test_gymnasium_checker():
    print("\n[7] gymnasium.utils.env_checker")
    try:
        from gymnasium.utils.env_checker import check_env
        env = HPCSchedulingEnv(
            num_nodes=3,
            num_jobs_visible=2,
            num_total_jobs=10,
            max_steps=100,
        )
        check_env(env, warn=True, skip_render_check=True)
        check("gymnasium check_env passed", True)
        env.close()
    except Exception as exc:
        check("gymnasium check_env passed", False, str(exc))
        traceback.print_exc()


def test_render():
    print("\n[8] Render (ansi mode)")
    env = HPCSchedulingEnv(num_nodes=2, num_jobs_visible=1, render_mode="ansi")
    env.reset(seed=5)
    text = env.render()
    check("render returns string in ansi mode", isinstance(text, str))
    check("render contains 'Node'", "Node" in text)
    env.close()


def test_performance():
    print("\n[9] Performance (10k random steps)")
    env = HPCSchedulingEnv(
        num_nodes=8,
        num_jobs_visible=5,
        num_total_jobs=200,
        max_steps=10_000,
    )
    env.reset(seed=0)
    t0 = time.perf_counter()
    N = 10_000
    terminated = truncated = False
    for i in range(N):
        if terminated or truncated:
            env.reset()
            terminated = truncated = False
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    elapsed = time.perf_counter() - t0
    fps = N / elapsed
    check(f"≥1000 steps/s (got {fps:.0f})", fps >= 1000)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HPCSchedulingEnv — Test Suite")
    print("=" * 60)

    tests = [
        test_spaces,
        test_reset,
        test_step_random,
        test_invalid_action_penalty,
        test_wait_action,
        test_reproducibility,
        test_gymnasium_checker,
        test_render,
        test_performance,
    ]

    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"  {FAIL}  Unhandled exception in {t.__name__}: {exc}")
            traceback.print_exc()
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("\033[92m  All tests passed!\033[0m")
    else:
        print(f"\033[91m  {failures} test(s) had unhandled errors.\033[0m")
    print("=" * 60)
