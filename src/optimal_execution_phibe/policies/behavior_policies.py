"""Simple behavior policies for offline dataset generation."""

from __future__ import annotations

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig


def twap_policy(obs: np.ndarray, cfg: GBMExecutionConfig) -> float:
    """Sell remaining inventory evenly over the remaining horizon."""

    t, q, _, _ = np.asarray(obs, dtype=np.float64)
    remaining_time = max(cfg.T - float(t), cfg.dt)
    return _clip_feasible(float(q) / remaining_time, float(q), cfg)


def noisy_twap_policy(
    obs: np.ndarray,
    cfg: GBMExecutionConfig,
    rng: np.random.Generator,
    noise_scale: float = 0.25,
) -> float:
    """TWAP with multiplicative Gaussian exploration."""

    base = twap_policy(obs, cfg)
    noisy_action = base * (1.0 + noise_scale * float(rng.normal()))
    return _clip_feasible(noisy_action, float(np.asarray(obs)[1]), cfg)


def front_loaded_policy(obs: np.ndarray, cfg: GBMExecutionConfig) -> float:
    """Sell faster early in the episode and taper later."""

    t, q, _, _ = np.asarray(obs, dtype=np.float64)
    progress = min(max(float(t) / cfg.T, 0.0), 1.0)
    multiplier = 1.75 - 0.75 * progress
    remaining_time = max(cfg.T - float(t), cfg.dt)
    return _clip_feasible(multiplier * float(q) / remaining_time, float(q), cfg)


def back_loaded_policy(obs: np.ndarray, cfg: GBMExecutionConfig) -> float:
    """Sell slowly early in the episode and faster later."""

    t, q, _, _ = np.asarray(obs, dtype=np.float64)
    progress = min(max(float(t) / cfg.T, 0.0), 1.0)
    multiplier = 0.45 + 1.55 * progress
    remaining_time = max(cfg.T - float(t), cfg.dt)
    return _clip_feasible(multiplier * float(q) / remaining_time, float(q), cfg)


def price_sensitive_policy(
    obs: np.ndarray,
    cfg: GBMExecutionConfig,
    price_sensitivity: float = 0.5,
) -> float:
    """Sell more when the current price is above the initial price."""

    t, q, _, log_s = np.asarray(obs, dtype=np.float64)
    remaining_time = max(cfg.T - float(t), cfg.dt)
    price_signal = float(log_s) - float(np.log(cfg.S0))
    multiplier = 1.0 + price_sensitivity * price_signal
    return _clip_feasible(multiplier * float(q) / remaining_time, float(q), cfg)


def random_feasible_policy(
    obs: np.ndarray,
    cfg: GBMExecutionConfig,
    rng: np.random.Generator,
) -> float:
    """Sample a random nonnegative feasible selling rate."""

    q = float(np.asarray(obs, dtype=np.float64)[1])
    return float(rng.uniform(0.0, _feasible_action_max(q, cfg)))


def _feasible_action_max(q: float, cfg: GBMExecutionConfig) -> float:
    return float(min(cfg.u_bar, max(q, 0.0) / cfg.dt))


def _clip_feasible(action: float, q: float, cfg: GBMExecutionConfig) -> float:
    return float(np.clip(action, 0.0, _feasible_action_max(q, cfg)))
