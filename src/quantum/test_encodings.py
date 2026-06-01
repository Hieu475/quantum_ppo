"""
test_encodings.py — Smoke Test for Generalized State Encoding
================================================================
Verifies that all three encoding strategies (angle, amplitude,
data_reuploading) can be instantiated and produce correct output shapes
for multiple environments.
"""

import sys
import traceback

import gymnasium as gym
import torch
import numpy as np

from config import Config
from state_encoder import PreEncodingNN
from quantum_actor import QuantumActor
from agent import HybridAgent


def test_pre_encoding_nn():
    """Test PreEncodingNN for various observation spaces."""
    print("\n" + "=" * 60)
    print(" TEST 1: PreEncodingNN")
    print("=" * 60)

    test_cases = [
        # (description, obs_space, n_qubits, encoding_type, expected_output_dim)
        ("Vector, identity (4→4)", gym.spaces.Box(-1, 1, (4,)), 4, "angle", 4),
        ("Vector, compress (8→4)", gym.spaces.Box(-1, 1, (8,)), 4, "angle", 4),
        ("Vector, expand (4→8)", gym.spaces.Box(-1, 1, (4,)), 8, "angle", 8),
        ("Vector, amplitude (8→2^3=8)", gym.spaces.Box(-1, 1, (8,)), 3, "amplitude", 8),
        ("Vector, amplitude compress (24→2^4=16)", gym.spaces.Box(-1, 1, (24,)), 4, "amplitude", 16),
        ("Image, grayscale (84×84)", gym.spaces.Box(0, 255, (84, 84), dtype=np.uint8), 4, "angle", 4),
        ("Image, RGB (84×84×3)", gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8), 4, "angle", 4),
    ]

    passed = 0
    for desc, obs_space, n_qubits, enc_type, expected_dim in test_cases:
        try:
            net = PreEncodingNN(obs_space, n_qubits, enc_type)
            sample = torch.tensor(obs_space.sample(), dtype=torch.float32)
            output = net(sample)

            # Check output dimension
            actual_dim = output.shape[-1]
            assert actual_dim == expected_dim, (
                f"Expected output dim {expected_dim}, got {actual_dim}"
            )

            # Check normalization range [-π, π]
            assert output.max() <= np.pi + 1e-5, f"Output exceeds π: {output.max()}"
            assert output.min() >= -np.pi - 1e-5, f"Output below -π: {output.min()}"

            print(f"   {desc}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
            passed += 1
        except Exception as e:
            print(f"   {desc}: {e}")
            traceback.print_exc()

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_quantum_actor():
    """Test QuantumActor with different encoding strategies."""
    print("\n" + "=" * 60)
    print(" TEST 2: QuantumActor (all encoding types)")
    print("=" * 60)

    test_cases = [
        # (env_name, n_qubits, encoding_type)
        ("CartPole-v1", 4, "angle"),
        ("CartPole-v1", 4, "data_reuploading"),
        ("CartPole-v1", 2, "amplitude"),   # 2^2=4 >= state_dim=4
    ]

    passed = 0
    for env_name, n_qubits, enc_type in test_cases:
        try:
            env = gym.make(env_name)
            obs_space = env.observation_space
            action_dim = env.action_space.n

            # Compute state_dim
            if len(obs_space.shape) == 1:
                state_dim = obs_space.shape[0]
            else:
                import math
                state_dim = math.prod(obs_space.shape)

            config = Config(
                env_name=env_name,
                state_dim=state_dim,
                action_dim=action_dim,
                n_qubits=n_qubits,
                n_layers=2,
                encoding_type=enc_type,
            )

            actor = QuantumActor(config, obs_space)

            # Single state forward pass
            state = torch.tensor(obs_space.sample(), dtype=torch.float32)
            logits = actor(state)
            assert logits.shape == (action_dim,), (
                f"Expected logits shape ({action_dim},), got {logits.shape}"
            )

            # Distribution test
            dist = actor.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Batch forward pass
            batch = torch.stack([
                torch.tensor(obs_space.sample(), dtype=torch.float32)
                for _ in range(3)
            ])
            batch_logits = actor(batch)
            assert batch_logits.shape == (3, action_dim), (
                f"Expected batch logits shape (3, {action_dim}), got {batch_logits.shape}"
            )

            params = sum(p.numel() for p in actor.parameters())
            print(f"   {env_name} + {enc_type} (q={n_qubits}): "
                  f"logits={logits.shape}, params={params}")
            passed += 1
            env.close()
        except Exception as e:
            print(f"   {env_name} + {enc_type} (q={n_qubits}): {e}")
            traceback.print_exc()

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_hybrid_agent():
    """Test full HybridAgent pipeline."""
    print("\n" + "=" * 60)
    print(" TEST 3: HybridAgent (full pipeline)")
    print("=" * 60)

    test_cases = [
        ("CartPole-v1", 4, "data_reuploading"),
        ("CartPole-v1", 4, "angle"),
        ("CartPole-v1", 2, "amplitude"),
    ]

    # Add LunarLander if available
    try:
        env_test = gym.make("LunarLander-v3")
        env_test.close()
        test_cases.extend([
            ("LunarLander-v3", 4, "angle"),
            ("LunarLander-v3", 4, "data_reuploading"),
            ("LunarLander-v3", 3, "amplitude"),  # 2^3=8 = state_dim
        ])
    except Exception:
        print("    LunarLander-v3 not available, skipping those tests")

    passed = 0
    for env_name, n_qubits, enc_type in test_cases:
        try:
            env = gym.make(env_name)
            obs_space = env.observation_space
            action_dim = env.action_space.n

            if len(obs_space.shape) == 1:
                state_dim = obs_space.shape[0]
            else:
                import math
                state_dim = math.prod(obs_space.shape)

            config = Config(
                env_name=env_name,
                state_dim=state_dim,
                action_dim=action_dim,
                n_qubits=n_qubits,
                n_layers=2,
                encoding_type=enc_type,
            )

            agent = HybridAgent(config, obs_space)

            # Test select_action
            state, _ = env.reset()
            action, log_prob, value = agent.select_action(state)
            assert isinstance(action, int), f"Action should be int, got {type(action)}"
            assert isinstance(log_prob, float), f"Log prob should be float"
            assert isinstance(value, float), f"Value should be float"

            # Test evaluate_actions (batch)
            states = torch.stack([
                torch.tensor(obs_space.sample(), dtype=torch.float32)
                for _ in range(4)
            ])
            actions = torch.tensor([0, 1, 0, 1])
            log_probs, entropy, values = agent.evaluate_actions(states, actions)
            assert log_probs.shape == (4,), f"log_probs shape: {log_probs.shape}"
            assert entropy.shape == (4,), f"entropy shape: {entropy.shape}"
            assert values.shape == (4,), f"values shape: {values.shape}"

            print(f"   {env_name} + {enc_type} (q={n_qubits}): "
                  f"action={action}, log_prob={log_prob:.3f}, value={value:.3f}")
            passed += 1
            env.close()
        except Exception as e:
            print(f"   {env_name} + {enc_type} (q={n_qubits}): {e}")
            traceback.print_exc()

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


if __name__ == "__main__":
    print(" Generalized State Encoding — Smoke Tests")
    print("=" * 60)

    results = []
    results.append(("PreEncodingNN", test_pre_encoding_nn()))
    results.append(("QuantumActor", test_quantum_actor()))
    results.append(("HybridAgent", test_hybrid_agent()))

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print(" All tests passed!")
    else:
        print("  Some tests failed. Check output above for details.")
        sys.exit(1)
