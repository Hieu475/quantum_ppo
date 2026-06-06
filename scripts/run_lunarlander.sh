#!/usr/bin/env bash
# =============================================================================
# run_lunarlander.sh — Train & Benchmark Quantum vs Classical PPO on LunarLander-v3
# =============================================================================
# Kết quả CartPole (outputs/runs/, outputs/benchmark_data/) được GIỮ NGUYÊN.
# LunarLander logs → outputs/runs_lunarlander/
# LunarLander benchmark → outputs/benchmark_data_lunarlander/
# =============================================================================

set -e

# Đường dẫn gốc
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$ROOT_DIR/outputs"

RUNS_DIR="$OUTPUTS_DIR/runs_lunarlander"
BENCH_DIR="$OUTPUTS_DIR/benchmark_data_lunarlander"
CKPT_DIR="$OUTPUTS_DIR/checkpoints_lunarlander"

mkdir -p "$RUNS_DIR" "$BENCH_DIR" "$CKPT_DIR"

ENV="LunarLander-v3"
TOTAL_STEPS=500000   # LunarLander cần nhiều hơn CartPole (~300k-500k để hội tụ)
N_QUBITS=6           # Giảm từ 8→6 để tăng tốc (~2x)
N_LAYERS=2           # Tăng layers để tăng biểu diễn
SEED=42

echo "============================================================"
echo " LunarLander-v3 — Quantum vs Classical PPO"
echo "============================================================"
echo "  Runs dir:    $RUNS_DIR"
echo "  Benchmark:   $BENCH_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  Timesteps:   $TOTAL_STEPS"
echo "============================================================"

# Cần thêm src vào PYTHONPATH để import đúng
export PYTHONPATH="$ROOT_DIR/src/quantum:$ROOT_DIR/src/classical${PYTHONPATH:+:$PYTHONPATH}"

# ── 1. Train Quantum PPO ──────────────────────────────────────────────────────
echo ""
echo "[1/3] Training Quantum (Hybrid) PPO on $ENV ..."
echo "    n_qubits=$N_QUBITS  n_layers=$N_LAYERS  total_timesteps=$TOTAL_STEPS"
python "$SCRIPT_DIR/train_quantum.py" \
    --env_name "$ENV" \
    --n_qubits $N_QUBITS \
    --n_layers $N_LAYERS \
    --encoding_type data_reuploading \
    --total_timesteps $TOTAL_STEPS \
    --rollout_steps 2048 \
    --actor_lr 1e-3 \
    --critic_lr 3e-4 \
    --entropy_coeff 0.01 \
    --seed $SEED \
    --log_dir "$RUNS_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    2>&1 | tee "$OUTPUTS_DIR/quantum_lunarlander.log"

echo ""
echo "[2/3] Training Classical PPO on $ENV ..."
python "$SCRIPT_DIR/train_classical.py" \
    --env_name "$ENV" \
    --total_timesteps $TOTAL_STEPS \
    --rollout_steps 2048 \
    --actor_lr 3e-4 \
    --critic_lr 1e-4 \
    --entropy_coeff 0.01 \
    --seed $SEED \
    --log_dir "$RUNS_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    2>&1 | tee "$OUTPUTS_DIR/classical_lunarlander.log"

# ── 3. Benchmark ──────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Generating benchmark analysis ..."
python "$SCRIPT_DIR/benchmark.py" \
    --runs_dir "$RUNS_DIR" \
    --output_dir "$BENCH_DIR" \
    --window 30

echo ""
echo "============================================================"
echo " DONE! Kết quả LunarLander lưu tại:"
echo "   $BENCH_DIR"
echo ""
echo " Kết quả CartPole GIỮ NGUYÊN tại:"
echo "   $OUTPUTS_DIR/benchmark_data"
echo ""
echo " Xem TensorBoard:"
echo "   tensorboard --logdir $RUNS_DIR"
echo "============================================================"
