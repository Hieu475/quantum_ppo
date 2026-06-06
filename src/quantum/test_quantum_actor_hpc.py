"""
test_quantum_actor_hpc.py — Kiểm tra QuantumActor với HPC env (obs_dim=21)
===========================================================================
Chạy từ thư mục gốc:
    python src/quantum/test_quantum_actor_hpc.py
"""

import sys
import math
import traceback

import torch
import numpy as np

sys.path.insert(0, "src/quantum")
sys.path.insert(0, ".")

from config import Config
from quantum_actor import QuantumActor
from envs import HPCSchedulingEnv


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
# Tạo env + config
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    return HPCSchedulingEnv(num_nodes=4, num_jobs_visible=3)  # obs_dim=21, action_dim=5


def make_config(n_qubits: int, compression_hidden=None) -> Config:
    env = make_env()
    cfg = Config(
        env_name="HPCSchedulingEnv",
        state_dim=env.obs_dim,       # 21
        action_dim=env.action_dim,   # 5
        action_type="discrete",
        n_qubits=n_qubits,
        n_layers=2,
        encoding_type="data_reuploading",
        compression_hidden=compression_hidden,
    )
    return cfg, env


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compression_layer_exists(n_qubits: int):
    print(f"\n[A] Kiểm tra lớp classical_compression  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)

    # Lớp classical_compression phải tồn tại
    check("classical_compression tồn tại", hasattr(actor, "classical_compression"))

    # Phải có Linear(21, n_qubits) là module đầu tiên
    first_linear = None
    for m in actor.classical_compression.modules():
        if isinstance(m, torch.nn.Linear):
            first_linear = m
            break

    if first_linear is not None:
        in_ok  = first_linear.in_features  == 21
        out_ok = first_linear.out_features == n_qubits
        check(
            f"Linear({first_linear.in_features} → {first_linear.out_features})",
            in_ok and out_ok,
            f"mong đợi Linear(21, {n_qubits})",
        )
    else:
        check("có Linear layer trong compression", False, "không tìm thấy")

    # LayerNorm phải có trong sequential
    has_ln = any(isinstance(m, torch.nn.LayerNorm)
                 for m in actor.classical_compression.modules())
    check("có LayerNorm trong compression head", has_ln)
    env.close()


def test_output_range_after_compression(n_qubits: int):
    print(f"\n[B] Kiểm tra output range sau Tanh×π  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)
    actor.eval()

    # Sinh 100 obs ngẫu nhiên, kiểm tra range
    obs = torch.rand(100, 21)
    with torch.no_grad():
        compressed = actor.classical_compression(obs)
        encoded    = torch.tanh(compressed) * math.pi

    min_val = encoded.min().item()
    max_val = encoded.max().item()
    check(
        f"encoded ∈ [-π, π]",
        min_val >= -math.pi - 1e-5 and max_val <= math.pi + 1e-5,
        f"min={min_val:.4f}, max={max_val:.4f}",
    )
    check(
        f"encoded shape = (100, {n_qubits})",
        encoded.shape == (100, n_qubits),
        str(tuple(encoded.shape)),
    )
    env.close()


def test_forward_single(n_qubits: int):
    print(f"\n[C] Forward pass — single obs  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)
    actor.eval()

    obs_np, _ = env.reset(seed=0)
    obs = torch.tensor(obs_np)

    with torch.no_grad():
        logits = actor(obs)

    check("output shape = (action_dim,)", logits.shape == (5,), str(tuple(logits.shape)))
    check("output là float", logits.dtype == torch.float32)
    check("output finite", torch.isfinite(logits).all().item())
    env.close()


def test_forward_batch(n_qubits: int):
    print(f"\n[D] Forward pass — batch obs  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)
    actor.eval()

    batch = torch.rand(4, 21)
    with torch.no_grad():
        logits = actor(batch)

    check("output shape = (4, action_dim)", logits.shape == (4, 5), str(tuple(logits.shape)))
    check("output finite", torch.isfinite(logits).all().item())
    env.close()


def test_action_sampling(n_qubits: int):
    print(f"\n[E] Lấy mẫu hành động  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)
    actor.eval()

    obs_np, _ = env.reset(seed=7)
    obs = torch.tensor(obs_np)

    with torch.no_grad():
        action, log_prob = actor.get_action_and_log_prob(obs)

    check("action là scalar tensor", action.shape == torch.Size([]))
    check("action trong [0, 4]", 0 <= action.item() <= 4, f"action={action.item()}")
    check("log_prob finite", torch.isfinite(log_prob).item())
    env.close()


def test_2layer_mlp_compression():
    print(f"\n[F] 2-layer MLP compression  (compression_hidden=16, n_qubits=6)")
    cfg, env = make_config(n_qubits=6, compression_hidden=16)
    actor = QuantumActor(cfg, env.observation_space)

    linears = [m for m in actor.classical_compression.modules()
               if isinstance(m, torch.nn.Linear)]
    check("có 2 Linear layers", len(linears) == 2, f"đếm được {len(linears)}")
    if len(linears) == 2:
        check("Linear[0]: 21 → 16", linears[0].in_features == 21 and linears[0].out_features == 16)
        check("Linear[1]: 16 → 6",  linears[1].in_features == 16 and linears[1].out_features == 6)

    obs = torch.rand(21)
    with torch.no_grad():
        logits = actor(obs)
    check("forward thành công với 2-layer MLP", logits.shape == (5,))
    env.close()


def test_param_count(n_qubits: int):
    print(f"\n[G] Đếm tham số  (n_qubits={n_qubits})")
    cfg, env = make_config(n_qubits)
    actor = QuantumActor(cfg, env.observation_space)

    compress_params = sum(p.numel() for p in actor.classical_compression.parameters())
    quantum_params  = actor.q_params.numel()
    post_params     = sum(p.numel() for p in actor.post_nn.parameters())
    total           = sum(p.numel() for p in actor.parameters())

    print(f"    classical_compression : {compress_params:5d} params  "
          f"  [Linear(21,{n_qubits}) + LayerNorm]")
    print(f"    q_params (VQC)        : {quantum_params:5d} params")
    print(f"    post_nn               : {post_params:5d} params")
    print(f"    TOTAL                 : {total:5d} params")
    check("compress_params > 0", compress_params > 0)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  QuantumActor + HPCSchedulingEnv — Classical Compression Test")
    print("=" * 65)

    failures = 0
    for n_q in [6, 8]:
        print(f"\n{'━'*65}")
        print(f"  n_qubits = {n_q}  →  Linear(21, {n_q})")
        print(f"{'━'*65}")
        for fn in [
            test_compression_layer_exists,
            test_output_range_after_compression,
            test_forward_single,
            test_forward_batch,
            test_action_sampling,
            test_param_count,
        ]:
            try:
                fn(n_q)
            except Exception as exc:
                print(f"  {FAIL}  Unhandled exception in {fn.__name__}: {exc}")
                traceback.print_exc()
                failures += 1

    # Test 2-layer MLP
    try:
        test_2layer_mlp_compression()
    except Exception as exc:
        print(f"  {FAIL}  Unhandled exception: {exc}")
        traceback.print_exc()
        failures += 1

    print("\n" + "=" * 65)
    if failures == 0:
        print("\033[92m  All tests passed!\033[0m")
    else:
        print(f"\033[91m  {failures} test(s) had errors.\033[0m")
    print("=" * 65)
