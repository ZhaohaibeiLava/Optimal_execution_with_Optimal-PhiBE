"""Tests for the GBM execution environment."""

from __future__ import annotations

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.envs import GBMExecutionEnv


def test_reset_returns_correct_initial_obs() -> None:
    cfg = GBMExecutionConfig()
    env = GBMExecutionEnv(cfg)

    obs, info = env.reset(seed=7)

    np.testing.assert_allclose(obs, np.array([0.0, cfg.Q, cfg.y0, np.log(cfg.S0)]))
    assert obs.dtype == np.float64
    assert info["step_count"] == 0


def test_q_decreases_by_u_dt_when_feasible() -> None:
    cfg = GBMExecutionConfig(sigma=0.0)
    env = GBMExecutionEnv(cfg)
    env.reset(seed=1)

    u = 1.25
    obs, _, _, _, info = env.step(u)

    expected_q = cfg.Q - u * cfg.dt
    assert np.isclose(obs[1], expected_q)
    assert np.isclose(info["q_next"], expected_q)


def test_action_clipping_prevents_negative_inventory() -> None:
    cfg = GBMExecutionConfig(dt=0.25, Q=1.0, u_bar=100.0)
    env = GBMExecutionEnv(cfg)
    env.reset(seed=1)

    obs, _, terminated, _, info = env.step(1000.0)

    assert obs[1] >= 0.0
    assert np.isclose(obs[1], 0.0)
    assert np.isclose(info["clipped_action"], cfg.Q / cfg.dt)
    assert terminated


def test_y_update_matches_exact_formula() -> None:
    cfg = GBMExecutionConfig(sigma=0.0, y0=0.3, rho=2.0, kappa=0.4, dt=0.1)
    env = GBMExecutionEnv(cfg)
    env.reset(seed=1)

    u = 1.5
    obs, _, _, _, info = env.step(u)

    decay = np.exp(-cfg.rho * cfg.dt)
    expected_y = decay * cfg.y0 + (cfg.kappa / cfg.rho) * (1.0 - decay) * u
    assert np.isclose(obs[2], expected_y)
    assert np.isclose(info["y_next"], expected_y)


def test_logS_update_is_deterministic_when_sigma_zero() -> None:
    cfg = GBMExecutionConfig(mu=0.1, sigma=0.0, dt=0.2)
    env = GBMExecutionEnv(cfg)
    env.reset(seed=1)

    obs, _, _, _, info = env.step(0.0)

    expected_log_s = np.log(cfg.S0) + cfg.mu * cfg.dt
    assert np.isclose(obs[3], expected_log_s)
    assert np.isclose(info["logS_next"], expected_log_s)


def test_final_terminal_penalty_is_applied_at_horizon() -> None:
    cfg = GBMExecutionConfig(T=0.1, dt=0.1, sigma=0.0, mu=0.0, lambda_risk=0.0, chi=7.0)
    env = GBMExecutionEnv(cfg)
    env.reset(seed=1)

    _, reward, terminated, truncated, info = env.step(0.0)

    assert not terminated
    assert truncated
    assert np.isclose(info["terminal_penalty"], cfg.chi * cfg.Q**2)
    assert np.isclose(reward, -cfg.chi * cfg.Q**2)


def test_no_transition_density_method_exists() -> None:
    env = GBMExecutionEnv(GBMExecutionConfig())

    assert not hasattr(env, "transition_density")
