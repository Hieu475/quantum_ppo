#!/usr/bin/env bash
# =============================================================================
# run_pendulum.sh — Train & Benchmark Quantum vs Classical PPO on Pendulum-v1
# =============================================================================
# Pendulum-v1: continuous action space (torque [-2, 2])
#   obs: Box(3,) — cos(θ), sin(θ), θ̇
#   act: Box(-2, 2, (1,)) — joint torque
#
# Kết quả CartPole và LunarLander được GIỮ NGUYÊN.
# Pendulum logs → outputs/runs_pendulum/
# Pendulum benchmark → outputs/benchmark_data_pendulum/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$ROOT_DIR/outputs"

RUNS_DIR="$OUTPUTS_DIR/runs_pendulum"
BENCH_DIR="$OUTPUTS_DIR/benchmark_data_pendulum"
CKPT_DIR="$OUTPUTS_DIR/checkpoints_pendulum"

mkdir -p "$RUNS_DIR" "$BENCH_DIR" "$CKPT_DIR"

ENV="Pendulum-v1"
TOTAL_STEPS=100000   # Pendulum đơn giản hơn, 100k bước là đủ
N_QUBITS=4           # Pendulum có 3 obs → 4 qubits là hợp lý
N_LAYERS=2           # 2 layers cho biểu cảm đủ
SEED=42

echo "============================================================"
echo " Pendulum-v1 — Quantum vs Classical PPO (Continuous Action)"
echo "============================================================"
echo "  Runs dir:    $RUNS_DIR"
echo "  Benchmark:   $BENCH_DIR"
echo "  Timesteps:   $TOTAL_STEPS"
echo "  n_qubits:    $N_QUBITS"
echo "  n_layers:    $N_LAYERS"
echo "  action type: continuous (Box(-2,2))"
echo "============================================================"

export PYTHONPATH="$ROOT_DIR/src/quantum:$ROOT_DIR/src/classical${PYTHONPATH:+:$PYTHONPATH}"

# ── 1. Train Quantum PPO ──────────────────────────────────────────────────────
echo ""
echo "[1/3] Training Quantum (Hybrid) PPO on $ENV ..."
python "$SCRIPT_DIR/train_quantum.py" \
    --env_name "$ENV" \
    --n_qubits $N_QUBITS \
    --n_layers $N_LAYERS \
    --encoding_type data_reuploading \
    --total_timesteps $TOTAL_STEPS \
    --rollout_steps 2048 \
    --actor_lr 3e-3 \
    --critic_lr 1e-3 \
    --entropy_coeff 0.005 \
    --seed $SEED \
    --log_dir "$RUNS_DIR" \
    2>&1 | tee "$OUTPUTS_DIR/quantum_pendulum.log"

# ── 2. Train Classical PPO ────────────────────────────────────────────────────
echo ""
echo "[2/3] Training Classical PPO on $ENV ..."
python "$SCRIPT_DIR/train_classical.py" \
    --env_name "$ENV" \
    --total_timesteps $TOTAL_STEPS \
    --rollout_steps 2048 \
    --actor_lr 3e-3 \
    --critic_lr 1e-3 \
    --entropy_coeff 0.005 \
    --seed $SEED \
    --log_dir "$RUNS_DIR" \
    2>&1 | tee "$OUTPUTS_DIR/classical_pendulum.log"

# ── 3. Benchmark ──────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Generating benchmark analysis ..."
python "$SCRIPT_DIR/benchmark.py" \
    --runs_dir "$RUNS_DIR" \
    --output_dir "$BENCH_DIR" \
    --window 20

echo ""
echo "============================================================"
echo " DONE! Kết quả Pendulum lưu tại:"
echo "   $BENCH_DIR"
echo ""
echo " Kết quả CartPole & LunarLander GIỮ NGUYÊN:"
echo "   $OUTPUTS_DIR/benchmark_data"
echo "   $OUTPUTS_DIR/benchmark_data_lunarlander"
echo ""
echo " Xem TensorBoard:"
echo "   tensorboard --logdir $RUNS_DIR"
echo "============================================================"
