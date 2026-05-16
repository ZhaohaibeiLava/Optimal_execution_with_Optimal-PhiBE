"""Tests for oracle Monte Carlo policy evaluation."""

from __future__ import annotations

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.evaluation import evaluate_policy_mc
from optimal_execution_phibe.policies import (
    back_loaded_policy,
    front_loaded_policy,
    random_feasible_policy,
    twap_policy,
)


def test_mc_returns_are_finite() -> None:
    cfg = GBMExecutionConfig(T=0.2, dt=0.05)

    result = evaluate_policy_mc(cfg, random_feasible_policy, n_episodes=20, seed=7)

    for value in result.values():
        assert np.isfinite(value)


def test_sigma_zero_deterministic_policy_repeated_evaluations_are_identical() -> None:
    cfg = GBMExecutionConfig(T=0.2, dt=0.05, sigma=0.0)

    first = evaluate_policy_mc(cfg, twap_policy, n_episodes=10, seed=11)
    second = evaluate_policy_mc(cfg, twap_policy, n_episodes=10, seed=999)

    assert first == second


def test_increasing_eta_weakly_reduces_aggressive_policy_relative_value() -> None:
    low_eta_cfg = GBMExecutionConfig(T=0.2, dt=0.05, sigma=0.0, eta=0.001)
    high_eta_cfg = GBMExecutionConfig(T=0.2, dt=0.05, sigma=0.0, eta=0.2)

    low_eta_gap = (
        evaluate_policy_mc(low_eta_cfg, front_loaded_policy, n_episodes=5, seed=3)[
            "mean_return"
        ]
        - evaluate_policy_mc(low_eta_cfg, back_loaded_policy, n_episodes=5, seed=3)[
            "mean_return"
        ]
    )
    high_eta_gap = (
        evaluate_policy_mc(high_eta_cfg, front_loaded_policy, n_episodes=5, seed=3)[
            "mean_return"
        ]
        - evaluate_policy_mc(high_eta_cfg, back_loaded_policy, n_episodes=5, seed=3)[
            "mean_return"
        ]
    )

    assert high_eta_gap <= low_eta_gap
