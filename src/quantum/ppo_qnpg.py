"""
ppo_qnpg.py — PPO with Quantum Natural Policy Gradient
=======================================================
Extends the base PPO-Clip algorithm with QNPG preconditioning for the
quantum actor's parameter updates. The key modification is replacing the
standard Adam update for quantum parameters with a QFIM-preconditioned
natural gradient step.

Architecture
------------
                          ┌─────────────────────────────────────┐
                          │           PPO-QNPG Update           │
                          │                                     │
  Policy Loss L(θ)   ─►  │  1. loss.backward() [autograd]     │
  Value Loss L(V)    ─►  │  2. For quantum params:            │
  Entropy Bonus H(π) ─►  │       g_nat = F(θ)^{-1} · g       │  ← QNPG
                          │       θ ← θ - η_q · g_nat         │
                          │  3. For classical params:           │
                          │       Adam step (critic + head)    │  ← Adam
                          └─────────────────────────────────────┘

Key Design Decisions
--------------------
1. **Hybrid optimization**: Only quantum circuit parameters (q_params,
   input_scaling) use QNPG. The classical compression head and post_nn
   layers use standard Adam, as they reside in Euclidean space.

2. **QFIM per PPO epoch**: QFIM is estimated once per rollout update
   (not per minibatch) to amortize the computational overhead.

3. **Graceful fallback**: If QNPG computation fails (e.g., numerical
   instabilities), the update transparently falls back to standard Adam.

4. **Ablation mode**: Setting `use_natural_grad=False` converts QNPG-PPO
   to standard PPO, enabling clean comparison for benchmarking.

Metrics Logged
--------------
All standard PPO metrics plus:
    - qnpg/euclidean_grad_norm: ‖∇L‖ before QFIM preconditioning
    - qnpg/natural_grad_norm:   ‖F⁻¹∇L‖ after preconditioning
    - qnpg/qfim_mean_diag:      Mean diagonal of estimated QFIM
    - qnpg/qfim_condition:      κ(F) = λ_max / λ_min (ill-conditioning)
    - qnpg/qfim_effective_dim:  #params with F_{ii} > threshold
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from config import Config
from agent import HybridAgent
from buffer import RolloutBuffer
from qfim import QFIMMode
from qnpg_optimizer import QNPGOptimizer


class PPO_QNPG:
    """
    PPO-Clip with Quantum Natural Policy Gradient for the quantum actor.

    Inherits the PPO-Clip surrogate objective but replaces the Adam update
    for quantum parameters with QFIM-preconditioned natural gradient descent.

    The QNPG update is applied to:
        - actor.q_params          (quantum circuit rotation angles)
        - actor.input_scaling     (trainable encoding scaling, if enabled)

    Standard Adam is used for:
        - actor.classical_compression   (linear compression head)
        - actor.post_nn                 (measurement-to-logits mapping)
        - critic.*                      (entire classical critic)

    Attributes:
        agent: Hybrid quantum-classical agent.
        config: Hyperparameter configuration (with QNPG extensions).
        qnpg_optimizer: Natural gradient optimizer for quantum params.
        classical_optimizer: Adam optimizer for classical params (actor head + critic).
        critic_optimizer: Adam optimizer for critic.
    """

    def __init__(self, agent: HybridAgent, config: Config) -> None:
        self.agent = agent
        self.config = config

        # ── Parse QNPG-specific config ────────────────────────────────────
        qfim_mode_str = getattr(config, "qnpg_qfim_mode", "diagonal")
        qfim_mode_map = {
            "diagonal": QFIMMode.DIAGONAL,
            "block_diag": QFIMMode.BLOCK_DIAGONAL,
            "full": QFIMMode.FULL,
        }
        qfim_mode = qfim_mode_map.get(qfim_mode_str, QFIMMode.DIAGONAL)

        qnpg_damping = getattr(config, "qnpg_damping", 1e-3)
        qnpg_samples = getattr(config, "qnpg_n_samples", 4)
        qnpg_block_size = getattr(config, "qnpg_block_size", 9)
        use_natural_grad = getattr(config, "qnpg_use_natural_grad", True)
        max_precondition_ratio = getattr(config, "qnpg_max_precondition_ratio", 10.0)
        warmup_steps = getattr(config, "qnpg_warmup_steps", 0)

        # ── QNPG optimizer for quantum parameters ─────────────────────────
        self.qnpg_optimizer = QNPGOptimizer(
            actor=agent.actor,
            lr=config.actor_lr,
            qfim_mode=qfim_mode,
            damping=qnpg_damping,
            n_qfim_samples=qnpg_samples,
            max_grad_norm=config.max_grad_norm,
            use_natural_grad=use_natural_grad,
            block_size=qnpg_block_size,
            max_precondition_ratio=max_precondition_ratio,
            warmup_steps=warmup_steps,
        )

        # ── Adam for classical actor head (compression + post_nn) ─────────
        # These parameters live in Euclidean space → standard Adam is optimal
        classical_actor_params = list(
            agent.actor.classical_compression.parameters()
        ) + list(agent.actor.post_nn.parameters())
        if hasattr(agent.actor, "log_std") and isinstance(
            agent.actor.log_std, nn.Parameter
        ):
            classical_actor_params.append(agent.actor.log_std)

        self.classical_actor_optimizer = torch.optim.Adam(
            classical_actor_params,
            lr=config.actor_lr,
        )

        # ── Adam for critic ────────────────────────────────────────────────
        self.critic_optimizer = torch.optim.Adam(
            agent.get_critic_params(), lr=config.critic_lr
        )

        # ── Expose optimizer for train.py compatibility ────────────────────
        # (train.py reads actor_optimizer.param_groups[0]["lr"] for logging)
        self.actor_optimizer = self.classical_actor_optimizer

    def update_learning_rate(self, global_step: int, total_timesteps: int) -> None:
        """Linearly decay learning rates to 0 over training."""
        frac = max(0.0, 1.0 - (global_step - 1.0) / total_timesteps)

        # Update classical actor Adam LR
        actor_lr = self.config.actor_lr * frac
        for pg in self.classical_actor_optimizer.param_groups:
            pg["lr"] = actor_lr

        # Update QNPG LR
        self.qnpg_optimizer.update_lr(self.config.actor_lr * frac)

        # Update critic LR
        critic_lr = self.config.critic_lr * frac
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = critic_lr

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform PPO-QNPG update over multiple epochs of minibatch updates.

        The QNPG update differs from standard PPO in the actor update step:
        instead of optimizer.step(), we:
          1. Compute loss and backprop (same as standard PPO)
          2. Apply QNPG step to quantum params (F^{-1} g update)
          3. Apply Adam step to classical actor params (compression + post_nn)

        Args:
            buffer: Rollout buffer with GAE advantages and returns.

        Returns:
            Extended metrics dictionary including standard PPO metrics
            plus QNPG-specific diagnostic metrics.
        """
        # ── Accumulators ──────────────────────────────────────────────────
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_advantage = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        # QNPG-specific accumulators
        total_euc_grad_norm = 0.0
        total_nat_grad_norm = 0.0
        total_qfim_mean_diag = 0.0
        total_qfim_condition = 0.0
        total_qfim_effective_dim = 0.0
        num_updates = 0

        # Get all states from buffer for QFIM estimation
        # buffer._states is set by compute_gae() call before update()
        all_states = buffer._states.clone().float()

        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_minibatches():
                # ── Evaluate current policy ───────────────────────────────
                new_log_probs, entropy, values = self.agent.evaluate_actions(
                    batch.states, batch.actions
                )

                # ── Importance sampling ratio ─────────────────────────────
                log_ratio = new_log_probs - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                # ── Approximate KL ────────────────────────────────────────
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                # ── Clipped policy loss ───────────────────────────────────
                advantages = batch.advantages
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                1.0 + self.config.clip_epsilon)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss ────────────────────────────────────────────
                value_loss = nn.functional.mse_loss(values, batch.returns)

                # ── Entropy bonus ─────────────────────────────────────────
                entropy_loss = entropy.mean()

                # ── Combined loss ─────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    - self.config.entropy_coeff * entropy_loss
                )

                # ── Clip fraction ─────────────────────────────────────────
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                    .float().mean().item()
                )

                # ════════════════════════════════════════════════════════
                # QNPG UPDATE — Key difference from standard PPO
                # ════════════════════════════════════════════════════════

                # Zero all gradients
                self.qnpg_optimizer.zero_grad()
                self.classical_actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Backward pass — populates .grad on all parameters
                loss.backward()

                # ── Classical parameters: standard Adam with grad clipping ──
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.get_critic_params(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent.actor.classical_compression.parameters())
                    + list(self.agent.actor.post_nn.parameters()),
                    self.config.max_grad_norm,
                )
                self.classical_actor_optimizer.step()
                self.critic_optimizer.step()

                # ── Quantum parameters: QNPG natural gradient update ────────
                # Uses F(θ)^{-1} · g instead of plain g
                qnpg_metrics = self.qnpg_optimizer.step(all_states)

                # ── Accumulate metrics ────────────────────────────────────
                total_actor_loss += policy_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_advantage += advantages.mean().item()
                actor_grad_norm = qnpg_metrics.get("natural_grad_norm", 0.0)
                total_actor_grad_norm += actor_grad_norm
                total_critic_grad_norm += (
                    critic_grad_norm.item()
                    if isinstance(critic_grad_norm, torch.Tensor)
                    else critic_grad_norm
                )
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction

                # QNPG-specific metrics
                total_euc_grad_norm += qnpg_metrics.get("euclidean_grad_norm", 0.0)
                total_nat_grad_norm += qnpg_metrics.get("natural_grad_norm", 0.0)
                total_qfim_mean_diag += qnpg_metrics.get("qfim_mean_diag", 0.0)
                total_qfim_condition += qnpg_metrics.get("qfim_condition", 0.0)
                total_qfim_effective_dim += float(
                    qnpg_metrics.get("qfim_effective_dim", 0)
                )
                num_updates += 1

        # ── Average metrics ────────────────────────────────────────────────
        n = max(num_updates, 1)
        return {
            # Standard PPO metrics (compatible with train.py logging)
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "avg_advantage": total_advantage / n,
            "actor_grad_norm": total_actor_grad_norm / n,
            "critic_grad_norm": total_critic_grad_norm / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_fraction / n,
            # QNPG-specific metrics
            "qnpg/euclidean_grad_norm": total_euc_grad_norm / n,
            "qnpg/natural_grad_norm": total_nat_grad_norm / n,
            "qnpg/qfim_mean_diag": total_qfim_mean_diag / n,
            "qnpg/qfim_condition": total_qfim_condition / n,
            "qnpg/qfim_effective_dim": total_qfim_effective_dim / n,
        }
