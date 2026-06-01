# Quantum PPO

A **Hybrid Quantum-Classical Proximal Policy Optimization (PPO)** agent implemented with [PennyLane](https://pennylane.ai/) and [PyTorch](https://pytorch.org/), benchmarked against a classical MLP baseline on CartPole-v1.

## Project Structure

```
quantum_ppo/
├── src/
│   ├── quantum/              # Quantum PPO source code
│   │   ├── quantum_actor.py  # Variational Quantum Circuit (VQC) actor
│   │   ├── state_encoder.py  # Classical pre-encoding NN
│   │   ├── critic.py         # Classical value network
│   │   ├── agent.py          # HybridAgent wrapper
│   │   ├── ppo.py            # PPO-Clip optimizer
│   │   ├── buffer.py         # Rollout buffer + GAE
│   │   ├── train.py          # Training loop
│   │   ├── config.py         # Hyperparameter config
│   │   └── utils.py          # Helpers & diagnostics
│   └── classical/            # Classical PPO source code
│       ├── classical_actor.py
│       ├── classical_agent.py
│       ├── critic.py
│       ├── buffer.py
│       ├── config.py
│       └── utils.py
├── scripts/
│   ├── train_quantum.py      # Entry point: train quantum agent
│   ├── train_classical.py    # Entry point: train classical baseline
│   ├── evaluate.py           # Evaluate & compare both agents
│   ├── benchmark.py          # Generate comparison figures
│   └── plot_comparison.py    # Additional plots
├── outputs/
│   ├── checkpoints/          # Saved model weights (.pt)
│   ├── runs/                 # TensorBoard logs
│   ├── benchmark_data/       # CSVs + figures
│   └── eval_videos/          # Recorded evaluation episodes
├── docs/
├── requirements.txt
└── README.md
```

## Quantum Encoding Strategies

| Strategy | Description | Gate depth |
|---|---|---|
| `angle` | RY(x_i) per qubit | O(q) |
| `amplitude` | Encode 2^q features into amplitudes | O(2^q) |
| `data_reuploading` | Re-encode at every layer (default) | O(q × L) |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Quantum PPO
```bash
# Default: CartPole-v1, data_reuploading, 4 qubits
python scripts/train_quantum.py

# Custom options
python scripts/train_quantum.py --env_name LunarLander-v3 --n_qubits 8
python scripts/train_quantum.py --encoding_type angle --n_qubits 4
```

### 3. Train Classical Baseline
```bash
python scripts/train_classical.py
```

### 4. Evaluate both agents
```bash
python scripts/evaluate.py --n_episodes 100
```

### 5. Generate benchmark figures
```bash
python scripts/benchmark.py
```

### 6. Monitor with TensorBoard
```bash
tensorboard --logdir outputs/runs
```

## Results (CartPole-v1)

| Metric | Quantum PPO | Classical PPO |
|---|---|---|
| Mean reward μ (100 eps) | 455.6 | **491.3** |
| Std σ | 83.9 | **22.5** |
| Perfect episodes (=500) | 71/100 | **78/100** |
| Convergence speed | ~38k steps | ~90k steps |
| Training FPS | ~23 | ~1,500 |
| Parameters | 42 | 30 |

> **Note:** GPU (lightning.gpu) is slower than CPU (lightning.qubit) for small qubit counts (< 20 qubits) due to data-transfer overhead. The code automatically selects the optimal device.

## Hyperparameters

Key defaults (see `src/quantum/config.py`):

| Parameter | Value |
|---|---|
| `n_qubits` | 4 |
| `n_layers` | 2 |
| `encoding_type` | `data_reuploading` |
| `actor_lr` | 5e-3 |
| `critic_lr` | 1e-3 |
| `total_timesteps` | 200,000 |
| `rollout_steps` | 2,048 |
| `ppo_epochs` | 10 |
| `gamma` | 0.99 |
| `clip_epsilon` | 0.2 |
