"""Tests for behavior policies."""

from __future__ import annotations

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.policies import (
    back_loaded_policy,
    front_loaded_policy,
    noisy_twap_policy,
    price_sensitive_policy,
    random_feasible_policy,
    twap_policy,
)


def test_policies_return_finite_scalar_actions() -> None:
    cfg = GBMExecutionConfig()
    obs = np.array([0.25, 0.7, 0.1, np.log(1.05)], dtype=np.float64)
    rng = np.random.default_rng(1)
    actions = [
        twap_policy(obs, cfg),
        noisy_twap_policy(obs, cfg, rng),
        front_loaded_policy(obs, cfg),
        back_loaded_policy(obs, cfg),
        price_sensitive_policy(obs, cfg),
        random_feasible_policy(obs, cfg, rng),
    ]

    for action in actions:
        assert np.isscalar(action)
        assert np.isfinite(action)


def test_actions_are_nonnegative_or_safely_clipped_by_env() -> None:
    cfg = GBMExecutionConfig()
    obs = np.array([0.95, 0.01, 0.0, np.log(cfg.S0) - 10.0], dtype=np.float64)
    rng = np.random.default_rng(2)
    actions = [
        twap_policy(obs, cfg),
        noisy_twap_policy(obs, cfg, rng),
        front_loaded_policy(obs, cfg),
        back_loaded_policy(obs, cfg),
        price_sensitive_policy(obs, cfg),
        random_feasible_policy(obs, cfg, rng),
    ]

    for action in actions:
        assert action >= 0.0


def test_random_feasible_policy_returns_values_within_feasible_range() -> None:
    cfg = GBMExecutionConfig(dt=0.1, Q=1.0)
    obs = np.array([0.0, 0.2, 0.0, np.log(cfg.S0)], dtype=np.float64)
    rng = np.random.default_rng(3)
    feasible_max = min(cfg.u_bar, obs[1] / cfg.dt)

    for _ in range(100):
        action = random_feasible_policy(obs, cfg, rng)
        assert 0.0 <= action <= feasible_max
