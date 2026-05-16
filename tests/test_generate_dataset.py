"""Tests for synthetic dataset generation."""

from __future__ import annotations

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.data import generate_dataset
from optimal_execution_phibe.data.generate_dataset import INFO_ARRAY_KEYS
from optimal_execution_phibe.policies import random_feasible_policy, twap_policy


def test_saved_dataset_contains_required_keys(tmp_path) -> None:
    dataset = _small_dataset()
    output_path = tmp_path / "dataset.npz"
    np.savez_compressed(output_path, **dataset)

    with np.load(output_path) as loaded:
        keys = set(loaded.files)

    required_keys = {
        "obs",
        "actions",
        "rewards",
        "next_obs",
        "terminated",
        "truncated",
        "episode_id",
        "step_id",
        *INFO_ARRAY_KEYS,
    }
    assert required_keys.issubset(keys)


def test_reward_decomposition_error_is_near_zero() -> None:
    dataset = _small_dataset()

    reconstructed_reward = (
        dataset["pnl"]
        - dataset["transient_cost"]
        - dataset["temporary_cost"]
        - dataset["risk_cost"]
        - dataset["terminal_penalty"]
    )

    np.testing.assert_allclose(dataset["rewards"], reconstructed_reward, atol=1e-12)


def test_generated_states_and_rewards_are_valid() -> None:
    dataset = _small_dataset()
    q = np.concatenate([dataset["obs"][:, 1], dataset["next_obs"][:, 1]])
    prices = np.concatenate([dataset["S"], dataset["S_next"]])

    assert np.all(q >= -1e-10)
    assert np.all(prices > 0.0)
    assert np.all(np.isfinite(dataset["rewards"]))


def _small_dataset() -> dict[str, np.ndarray]:
    cfg = GBMExecutionConfig(T=0.2, dt=0.05, seed=17)
    return generate_dataset(
        cfg=cfg,
        n_episodes=5,
        policy_mixture=[twap_policy, random_feasible_policy],
        seed=cfg.seed,
    )
