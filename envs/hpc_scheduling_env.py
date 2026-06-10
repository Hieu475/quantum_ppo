"""
hpc_scheduling_env.py — HPC Resource Allocation Scheduling Environment
=======================================================================
A Gymnasium-compatible environment modelling an HPC cluster scheduler as an MDP.

Problem formulation
-------------------
At each discrete time-step the agent sees the current state of N compute nodes
and the first K jobs waiting in the queue.  It must decide:
  • Which node to dispatch the head-of-queue job to (actions 0 … N-1), or
  • Wait one time-step so running jobs can finish and free resources (action N).

State space
-----------
Flat float32 vector of length  N * 3 + K * 3:
  • Per-node features  : [cpu_used_ratio, ram_used_ratio, max_time_remaining_ratio]
  • Per-job features   : [cpu_req_ratio,  ram_req_ratio,  estimated_duration_ratio]
All values are normalised to [0, 1].

Action space
------------
Discrete(N + 1):
  0 … N-1  →  schedule head-of-queue job on that node
  N        →  wait one time-step (no dispatch)

Reward shaping
--------------
  R_t = α · avg_cpu_utilisation
       - β · (queue_length / max_queue_size)   # penalise backlog
       + γ · (job dispatched successfully)       # dispatch bonus
       - δ · (invalid dispatch attempted)        # hard penalty

Parameters α, β, γ, δ are configurable via the constructor.

Compatibility note for Quantum PPO
-----------------------------------
The observation dimension can be large (e.g. 3*N + 3*K).  Before feeding into
the quantum circuit, compress with a Classical Linear head inside
state_encoder.PreEncodingNN down to n_qubits features.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Internal data structures

class Job:
    """A single HPC workload unit waiting in or running on the cluster."""

    def __init__(
        self,
        job_id: int,
        cpu_req: float,
        ram_req: float,
        duration: float,
    ) -> None:
        """
        Args:
            job_id:   Unique identifier.
            cpu_req:  CPU fraction required, in [0, 1].
            ram_req:  RAM fraction required, in [0, 1].
            duration: Estimated run time in time-steps.
        """
        self.job_id = job_id
        self.cpu_req = float(np.clip(cpu_req, 0.0, 1.0))
        self.ram_req = float(np.clip(ram_req, 0.0, 1.0))
        self.duration = max(1.0, float(duration))
        self.time_left = self.duration  # decremented each tick

    def tick(self) -> bool:
        """Advance one time-step.  Returns True when job finishes."""
        self.time_left = max(0.0, self.time_left - 1.0)
        return self.time_left <= 0.0

    def __repr__(self) -> str:
        return (
            f"Job(id={self.job_id}, cpu={self.cpu_req:.2f}, "
            f"ram={self.ram_req:.2f}, left={self.time_left:.1f})"
        )


class Node:
    """A single compute node in the HPC cluster."""

    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.cpu_used: float = 0.0   # in [0, 1]
        self.ram_used: float = 0.0   # in [0, 1]
        self.running_jobs: List[Job] = []

    # Resource checks

    def can_fit(self, job: Job) -> bool:
        """Return True if the node has enough CPU and RAM for the job."""
        return (
            self.cpu_used + job.cpu_req <= 1.0 + 1e-6
            and self.ram_used + job.ram_req <= 1.0 + 1e-6
        )

    def assign(self, job: Job) -> None:
        """Schedule a job on this node (no capacity check — call can_fit first)."""
        self.cpu_used = min(1.0, self.cpu_used + job.cpu_req)
        self.ram_used = min(1.0, self.ram_used + job.ram_req)
        self.running_jobs.append(job)

    # Time advancement

    def tick(self) -> int:
        """
        Advance one time-step: decrement all job timers and evict finished jobs.

        Returns:
            Number of jobs that finished during this tick.
        """
        finished_count = 0
        still_running: List[Job] = []
        for job in self.running_jobs:
            done = job.tick()
            if done:
                self.cpu_used = max(0.0, self.cpu_used - job.cpu_req)
                self.ram_used = max(0.0, self.ram_used - job.ram_req)
                finished_count += 1
            else:
                still_running.append(job)
        self.running_jobs = still_running
        return finished_count

    # Observation features

    @property
    def max_time_remaining(self) -> float:
        """Longest remaining run time among all active jobs (0 if idle)."""
        if not self.running_jobs:
            return 0.0
        return max(j.time_left for j in self.running_jobs)

    def reset(self) -> None:
        """Clear all jobs and resource counters."""
        self.cpu_used = 0.0
        self.ram_used = 0.0
        self.running_jobs.clear()

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, cpu={self.cpu_used:.2f}, "
            f"ram={self.ram_used:.2f}, jobs={len(self.running_jobs)})"
        )

# Main environment

class HPCSchedulingEnv(gym.Env):
    """
    HPC Cluster Scheduling Environment (Gymnasium API).

    The agent acts as a centralized job scheduler that dispatches workloads
    from a job queue to a fixed pool of compute nodes, maximising resource
    utilisation while minimising waiting times.

    Observation:
        numpy float32 array of shape (obs_dim,), where
        obs_dim = num_nodes * 3 + num_jobs_visible * 3.

    Action:
        Integer in {0, …, num_nodes}.
        0 … num_nodes-1 : schedule head-of-queue job on the given node.
        num_nodes        : wait one time-step.

    Episode termination:
        • All jobs in the initial queue have been completed (``terminated``).
        • OR ``max_steps`` time-steps have elapsed (``truncated``).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        # Cluster topology
        num_nodes: int = 4,
        # Queue visibility window
        num_jobs_visible: int = 3,
        # Episode workload size
        num_total_jobs: int = 20,
        max_steps: int = 500,
        # Job generation parameters
        cpu_req_range: Tuple[float, float] = (0.1, 0.5),
        ram_req_range: Tuple[float, float] = (0.1, 0.5),
        duration_range: Tuple[float, float] = (2.0, 15.0),
        # Reward shaping coefficients
        alpha: float = 1.0,   # utilisation reward weight
        beta: float = 0.3,    # queue-length penalty weight
        gamma: float = 5.0,   # successful dispatch bonus
        delta: float = 10.0,  # invalid dispatch penalty
        # Misc
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # ── Cluster / episode parameters ──────────────────────────────────
        self.num_nodes = num_nodes
        self.num_jobs_visible = num_jobs_visible
        self.num_total_jobs = num_total_jobs
        self.max_steps = max_steps

        # ── Job generation parameters ─────────────────────────────────────
        self.cpu_req_range = cpu_req_range
        self.ram_req_range = ram_req_range
        self.duration_range = duration_range

        # ── Reward coefficients ───────────────────────────────────────────
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # ── Rendering ────────────────────────────────────────────────────
        self.render_mode = render_mode

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Action: choose a node (0..N-1) or wait (N)
        self.action_space = spaces.Discrete(self.num_nodes + 1)

        # Observation: [N × (cpu_used, ram_used, max_time_norm)]
        #             + [K × (cpu_req, ram_req, duration_norm)]
        # All values in [0, 1].
        obs_dim = self.num_nodes * 3 + self.num_jobs_visible * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── Internal state (initialised in reset()) ───────────────────────
        self.nodes: List[Node] = []
        self.job_queue: deque[Job] = deque()
        self.completed_jobs: int = 0
        self.total_jobs_created: int = 0
        self._step_count: int = 0
        self._job_id_counter: int = 0

        # Seed the RNG
        self._rng = np.random.default_rng(seed)

        # Normalisation constant for job durations (used in observations)
        self._max_duration = self.duration_range[1]

    # Gymnasium API — reset

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Returns:
            observation: Initial flattened state vector.
            info: Auxiliary diagnostic information.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset cluster
        self.nodes = [Node(i) for i in range(self.num_nodes)]

        # Generate a fresh job workload for this episode
        self._job_id_counter = 0
        self.job_queue = deque(self._generate_jobs(self.num_total_jobs))
        self.completed_jobs = 0
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # Gymnasium API — step

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply action and advance the simulation by one time-step.

        Args:
            action: Integer in {0, …, num_nodes}.

        Returns:
            observation: New state vector.
            reward:      Shaped scalar reward.
            terminated:  True when all jobs completed.
            truncated:   True when max_steps reached.
            info:        Diagnostic dictionary.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        dispatched = False

        # ── Attempt dispatch ───────────────────────────────────────────────
        if len(self.job_queue) > 0:
            head_job = self.job_queue[0]

            if action < self.num_nodes:
                # Agent picks a specific node
                target_node = self.nodes[action]
                if target_node.can_fit(head_job):
                    # ✔ Valid dispatch
                    target_node.assign(head_job)
                    self.job_queue.popleft()
                    dispatched = True
                    reward += self.gamma          # dispatch bonus
                else:
                    # ✘ Invalid dispatch — node is full
                    reward -= self.delta          # hard penalty
                    # Treat as implicit wait (do NOT pop from queue)
            else:
                # Agent explicitly waits
                reward -= 0.5                    # tiny wait discouragement

        # ── Advance simulation clock ───────────────────────────────────────
        finished_this_tick = self._tick_all_nodes()
        self.completed_jobs += finished_this_tick

        # ── Utilisation reward (average CPU across all nodes) ──────────────
        avg_cpu = np.mean([n.cpu_used for n in self.nodes])
        reward += self.alpha * avg_cpu

        # ── Queue backlog penalty ──────────────────────────────────────────
        queue_ratio = len(self.job_queue) / max(1, self.num_total_jobs)
        reward -= self.beta * queue_ratio

        self._step_count += 1

        # ── Termination conditions ─────────────────────────────────────────
        all_dispatched_and_done = (
            len(self.job_queue) == 0
            and all(len(n.running_jobs) == 0 for n in self.nodes)
        )
        terminated = all_dispatched_and_done
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["dispatched"] = dispatched
        info["finished_jobs"] = finished_this_tick

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # Rendering

    def render(self) -> Optional[str]:
        """Print or return a human-readable cluster status string."""
        lines = [
            f"\n{'─'*60}",
            f"  Step {self._step_count:4d} | Queue: {len(self.job_queue):3d} jobs"
            f" | Completed: {self.completed_jobs:3d}",
            f"{'─'*60}",
        ]
        for node in self.nodes:
            bar_cpu = "█" * int(node.cpu_used * 20)
            bar_ram = "█" * int(node.ram_used * 20)
            lines.append(
                f"  Node {node.node_id}: CPU [{bar_cpu:<20}] {node.cpu_used:.0%}"
                f"  RAM [{bar_ram:<20}] {node.ram_used:.0%}"
                f"  ({len(node.running_jobs)} jobs)"
            )
        if self.job_queue:
            head = self.job_queue[0]
            lines.append(
                f"\n  Next job → cpu={head.cpu_req:.2f}, "
                f"ram={head.ram_req:.2f}, dur={head.duration:.1f}"
            )
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text  # "ansi" mode

    # Private helpers

    def _generate_jobs(self, n: int) -> List[Job]:
        """
        Randomly sample a list of n Job objects.

        Uses the environment's internal RNG so episodes are reproducible
        when ``reset(seed=...)`` is called.
        """
        jobs: List[Job] = []
        for _ in range(n):
            cpu = float(
                self._rng.uniform(self.cpu_req_range[0], self.cpu_req_range[1])
            )
            ram = float(
                self._rng.uniform(self.ram_req_range[0], self.ram_req_range[1])
            )
            dur = float(
                self._rng.uniform(self.duration_range[0], self.duration_range[1])
            )
            jobs.append(Job(self._job_id_counter, cpu, ram, dur))
            self._job_id_counter += 1
        return jobs

    def _tick_all_nodes(self) -> int:
        """Advance all nodes by one tick.  Returns total finished job count."""
        total_finished = 0
        for node in self.nodes:
            total_finished += node.tick()
        return total_finished

    def _get_obs(self) -> np.ndarray:
        """
        Build the flattened observation vector.

        Layout:
          [node_0_cpu, node_0_ram, node_0_time,
           node_1_cpu, ...,
           job_0_cpu,  job_0_ram,  job_0_dur,
           job_1_cpu,  ...,
           (zeros if fewer than K jobs in queue)]
        """
        node_features: List[float] = []
        for node in self.nodes:
            node_features.extend([
                node.cpu_used,                                        # [0, 1]
                node.ram_used,                                        # [0, 1]
                node.max_time_remaining / self._max_duration,         # [0, 1]
            ])

        job_features: List[float] = []
        queue_list = list(self.job_queue)
        for i in range(self.num_jobs_visible):
            if i < len(queue_list):
                job = queue_list[i]
                job_features.extend([
                    job.cpu_req,                                      # [0, 1]
                    job.ram_req,                                      # [0, 1]
                    min(job.duration / self._max_duration, 1.0),      # [0, 1]
                ])
            else:
                # Pad with zeros when queue has fewer than K jobs
                job_features.extend([0.0, 0.0, 0.0])

        obs = np.array(node_features + job_features, dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic info dict (not used for learning, only monitoring)."""
        return {
            "step": self._step_count,
            "queue_length": len(self.job_queue),
            "completed_jobs": self.completed_jobs,
            "avg_cpu_utilisation": float(
                np.mean([n.cpu_used for n in self.nodes])
            ),
            "avg_ram_utilisation": float(
                np.mean([n.ram_used for n in self.nodes])
            ),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "cpu_used": n.cpu_used,
                    "ram_used": n.ram_used,
                    "running_jobs": len(n.running_jobs),
                }
                for n in self.nodes
            ],
        }

    # Utility properties (useful for downstream code / config)

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimension."""
        return int(np.prod(self.observation_space.shape))

    @property
    def action_dim(self) -> int:
        """Number of discrete actions."""
        return int(self.action_space.n)
