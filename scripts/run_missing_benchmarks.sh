#!/usr/bin/env bash
# =============================================================================
# run_missing_benchmarks.sh — Chạy phần còn thiếu trong benchmark
# =============================================================================
# Tình trạng hiện tại:
#   CartPole   → ✅ Quantum + Classical + Benchmark đầy đủ
#   LunarLander → ⚠️  Quantum ✅ | Classical ❌ | Benchmark ❌
#   Pendulum    → ⚠️  Quantum ✅ | Classical ❌ | Benchmark ❌ (quantum only)
#
# Script này sẽ:
#   [1] Chạy Classical PPO trên LunarLander-v3 (300k steps)
#   [2] Chạy benchmark LunarLander (Quantum vs Classical)
#   [3] Chạy Classical PPO trên Pendulum-v1 (100k steps)
#   [4] Chạy benchmark Pendulum (Quantum vs Classical)
#   [5] In tổng kết kết quả tất cả môi trường
#
# Usage:
#   bash scripts/run_missing_benchmarks.sh
#   bash scripts/run_missing_benchmarks.sh --skip-lunarlander   # Bỏ qua LunarLander
#   bash scripts/run_missing_benchmarks.sh --skip-pendulum      # Bỏ qua Pendulum
#   bash scripts/run_missing_benchmarks.sh --lunarlander-only   # Chỉ LunarLander
#   bash scripts/run_missing_benchmarks.sh --pendulum-only      # Chỉ Pendulum
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$ROOT_DIR/outputs"

# ── Màu sắc terminal ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Parse arguments ───────────────────────────────────────────────────────────
RUN_LUNARLANDER=true
RUN_PENDULUM=true

for arg in "$@"; do
    case $arg in
        --skip-lunarlander)   RUN_LUNARLANDER=false ;;
        --skip-pendulum)      RUN_PENDULUM=false ;;
        --lunarlander-only)   RUN_PENDULUM=false ;;
        --pendulum-only)      RUN_LUNARLANDER=false ;;
    esac
done

# ── Đường dẫn ─────────────────────────────────────────────────────────────────
RUNS_LUNARLANDER="$OUTPUTS_DIR/runs_lunarlander"
RUNS_PENDULUM="$OUTPUTS_DIR/runs_pendulum"
BENCH_LUNARLANDER="$OUTPUTS_DIR/benchmark_data_lunarlander"
BENCH_PENDULUM="$OUTPUTS_DIR/benchmark_data_pendulum"
CKPT_LUNARLANDER="$OUTPUTS_DIR/checkpoints_lunarlander"
CKPT_PENDULUM="$OUTPUTS_DIR/checkpoints_pendulum"

mkdir -p "$RUNS_LUNARLANDER" "$RUNS_PENDULUM"
mkdir -p "$BENCH_LUNARLANDER" "$BENCH_PENDULUM"
mkdir -p "$CKPT_LUNARLANDER" "$CKPT_PENDULUM"

# ── PYTHONPATH ─────────────────────────────────────────────────────────────────
export PYTHONPATH="$ROOT_DIR/src/quantum:$ROOT_DIR/src/classical${PYTHONPATH:+:$PYTHONPATH}"

OVERALL_START=$(date +%s)

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Quantum PPO — Benchmark Runner (Missing Parts)       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  CartPole   : ✅ Đã có đầy đủ kết quả                       ║"
echo "║  LunarLander: $([ "$RUN_LUNARLANDER" = true ] && echo "🔄 Sẽ chạy Classical + Benchmark     " || echo "⏭️  Bỏ qua theo yêu cầu              ")║"
echo "║  Pendulum   : $([ "$RUN_PENDULUM" = true ] && echo "🔄 Sẽ chạy Classical + Benchmark     " || echo "⏭️  Bỏ qua theo yêu cầu              ")║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1: LUNARLANDER-V3
# ══════════════════════════════════════════════════════════════════════════════
if [ "$RUN_LUNARLANDER" = true ]; then
    echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  PHẦN 1/2 — LunarLander-v3${NC}"
    echo -e "${CYAN}  Quantum: ✅ Đã hoàn thành (150k steps)${NC}"
    echo -e "${CYAN}  Classical: 🔄 Bắt đầu chạy...${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}\n"

    LL_CLASSICAL_LOG="$OUTPUTS_DIR/classical_lunarlander_new.log"

    echo -e "${YELLOW}[1a/4] Training Classical PPO on LunarLander-v3 (300k steps)...${NC}"
    echo "       Log: $LL_CLASSICAL_LOG"
    echo "       Có thể mất 10-20 phút..."
    echo ""

    python "$SCRIPT_DIR/train_classical.py" \
        --env_name "LunarLander-v3" \
        --total_timesteps 300000 \
        --rollout_steps 2048 \
        --actor_lr 3e-4 \
        --critic_lr 1e-4 \
        --entropy_coeff 0.01 \
        --critic_hidden 128 \
        --seed 42 \
        --log_dir "$RUNS_LUNARLANDER" \
        --checkpoint_dir "$CKPT_LUNARLANDER" \
        2>&1 | tee "$LL_CLASSICAL_LOG"

    echo ""
    echo -e "${GREEN}✅ Classical PPO LunarLander hoàn thành!${NC}"

    echo ""
    echo -e "${YELLOW}[1b/4] Generating LunarLander benchmark (Quantum vs Classical)...${NC}"

    python "$SCRIPT_DIR/benchmark.py" \
        --runs_dir "$RUNS_LUNARLANDER" \
        --output_dir "$BENCH_LUNARLANDER" \
        --window 30

    echo -e "${GREEN}✅ LunarLander benchmark hoàn thành!${NC}"
    echo -e "   📊 Kết quả: ${BOLD}$BENCH_LUNARLANDER${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2: PENDULUM-V1
# ══════════════════════════════════════════════════════════════════════════════
if [ "$RUN_PENDULUM" = true ]; then
    echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  PHẦN 2/2 — Pendulum-v1${NC}"
    echo -e "${CYAN}  Quantum: ✅ Đã hoàn thành (100k steps)${NC}"
    echo -e "${CYAN}  Classical: 🔄 Bắt đầu chạy...${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}\n"

    PD_CLASSICAL_LOG="$OUTPUTS_DIR/classical_pendulum_new.log"

    echo -e "${YELLOW}[2a/4] Training Classical PPO on Pendulum-v1 (100k steps)...${NC}"
    echo "       Log: $PD_CLASSICAL_LOG"
    echo "       Có thể mất 3-5 phút..."
    echo ""

    python "$SCRIPT_DIR/train_classical.py" \
        --env_name "Pendulum-v1" \
        --total_timesteps 100000 \
        --rollout_steps 2048 \
        --actor_lr 3e-3 \
        --critic_lr 1e-3 \
        --entropy_coeff 0.005 \
        --critic_hidden 64 \
        --seed 42 \
        --log_dir "$RUNS_PENDULUM" \
        --checkpoint_dir "$CKPT_PENDULUM" \
        2>&1 | tee "$PD_CLASSICAL_LOG"

    echo ""
    echo -e "${GREEN}✅ Classical PPO Pendulum hoàn thành!${NC}"

    echo ""
    echo -e "${YELLOW}[2b/4] Generating Pendulum benchmark (Quantum vs Classical)...${NC}"

    python "$SCRIPT_DIR/benchmark.py" \
        --runs_dir "$RUNS_PENDULUM" \
        --output_dir "$BENCH_PENDULUM" \
        --window 20

    echo -e "${GREEN}✅ Pendulum benchmark hoàn thành!${NC}"
    echo -e "   📊 Kết quả: ${BOLD}$BENCH_PENDULUM${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TỔNG KẾT
# ══════════════════════════════════════════════════════════════════════════════
OVERALL_END=$(date +%s)
ELAPSED=$(( OVERALL_END - OVERALL_START ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🎉 BENCHMARK HOÀN TẤT!                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Tổng thời gian: %3d phút %2d giây                           ║\n" $MINUTES $SECONDS_REM
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Kết quả theo môi trường:                                    ║"
echo "║                                                              ║"
echo "║  📁 CartPole   : outputs/benchmark_data/                     ║"
echo "║  📁 LunarLander: outputs/benchmark_data_lunarlander/         ║"
echo "║  📁 Pendulum   : outputs/benchmark_data_pendulum/            ║"
echo "║                                                              ║"
echo "║  📈 TensorBoard (tất cả):                                    ║"
echo "║     tensorboard --logdir outputs/runs                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── In tóm tắt kết quả từ benchmark_report.json ───────────────────────────────
echo -e "${BOLD}📊 Tóm tắt kết quả:${NC}"
echo ""

print_report() {
    local name="$1"
    local report_file="$2"
    if [ -f "$report_file" ]; then
        echo -e "${BOLD}${CYAN}═══ $name ═══${NC}"
        python3 -c "
import json, sys
try:
    with open('$report_file') as f:
        d = json.load(f)
    q = d.get('quantum', {})
    c = d.get('classical', {})
    print(f\"  {'Metric':<30} {'Quantum PPO':>15} {'Classical PPO':>15}\")
    print(f\"  {'-'*60}\")
    if q and c:
        print(f\"  {'Best reward':<30} {q.get('best_reward', 'N/A'):>15.1f} {c.get('best_reward', 'N/A'):>15.1f}\")
        print(f\"  {'Final avg reward (100 ep)':<30} {q.get('final_mean_100', 'N/A'):>15.2f} {c.get('final_mean_100', 'N/A'):>15.2f}\")
        print(f\"  {'Std dev (100 ep)':<30} {q.get('final_std_100', 'N/A'):>15.2f} {c.get('final_std_100', 'N/A'):>15.2f}\")
        print(f\"  {'Total episodes':<30} {q.get('total_episodes', 'N/A'):>15} {c.get('total_episodes', 'N/A'):>15}\")
        avg_fps_q = q.get('avg_fps', None)
        avg_fps_c = c.get('avg_fps', None)
        fps_q_str = f'{avg_fps_q:.0f}' if avg_fps_q else 'N/A'
        fps_c_str = f'{avg_fps_c:.0f}' if avg_fps_c else 'N/A'
        print(f\"  {'Avg FPS':<30} {fps_q_str:>15} {fps_c_str:>15}\")
    elif q:
        print(f\"  Quantum only: best={q.get('best_reward', 'N/A'):.1f}, final_avg={q.get('final_mean_100', 'N/A'):.2f}\")
        print(f\"  Classical: No data (chưa chạy)\")
    else:
        print('  No data available')
except Exception as e:
    print(f'  Error reading report: {e}')
"
        echo ""
    fi
}

print_report "CartPole-v1" "$OUTPUTS_DIR/benchmark_data/benchmark_report.json"
print_report "LunarLander-v3" "$OUTPUTS_DIR/benchmark_data_lunarlander/benchmark_report.json"
print_report "Pendulum-v1" "$OUTPUTS_DIR/benchmark_data_pendulum/benchmark_report.json"

echo -e "${BOLD}Xem hình ảnh benchmark tại các thư mục outputs/benchmark_data_*/${NC}"
