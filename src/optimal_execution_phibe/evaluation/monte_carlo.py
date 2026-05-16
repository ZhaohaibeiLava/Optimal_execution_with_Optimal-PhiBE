"""Oracle Monte Carlo policy evaluation for the GBM execution environment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.envs import GBMExecutionEnv

PolicyFn = Callable[..., float]


def evaluate_policy_mc(
    cfg: GBMExecutionConfig,
    policy_fn: PolicyFn,
    n_episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate a policy by Monte Carlo rollouts in the simulator."""

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")

    rng = np.random.default_rng(seed)
    returns = np.empty(n_episodes, dtype=np.float64)
    terminal_inventory = np.empty(n_episodes, dtype=np.float64)
    total_pnl = np.empty(n_episodes, dtype=np.float64)
    total_temporary_cost = np.empty(n_episodes, dtype=np.float64)
    total_transient_cost = np.empty(n_episodes, dtype=np.float64)
    total_risk_cost = np.empty(n_episodes, dtype=np.float64)
    total_terminal_penalty = np.empty(n_episodes, dtype=np.float64)

    for episode_id in range(n_episodes):
        env = GBMExecutionEnv(cfg)
        obs, _ = env.reset(seed=int(rng.integers(0, np.iinfo(np.uint32).max)))
        done = False
        episode_return = 0.0
        episode_pnl = 0.0
        episode_temporary_cost = 0.0
        episode_transient_cost = 0.0
        episode_risk_cost = 0.0
        episode_terminal_penalty = 0.0

        while not done:
            action = _call_policy(policy_fn, obs, cfg, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_pnl += float(info["pnl"])
            episode_temporary_cost += float(info["temporary_cost"])
            episode_transient_cost += float(info["transient_cost"])
            episode_risk_cost += float(info["risk_cost"])
            episode_terminal_penalty += float(info["terminal_penalty"])
            done = terminated or truncated

        returns[episode_id] = episode_return
        terminal_inventory[episode_id] = float(obs[1])
        total_pnl[episode_id] = episode_pnl
        total_temporary_cost[episode_id] = episode_temporary_cost
        total_transient_cost[episode_id] = episode_transient_cost
        total_risk_cost[episode_id] = episode_risk_cost
        total_terminal_penalty[episode_id] = episode_terminal_penalty

    std_return = float(returns.std(ddof=1)) if n_episodes > 1 else 0.0
    return {
        "mean_return": float(returns.mean()),
        "std_return": std_return,
        "stderr_return": float(std_return / np.sqrt(n_episodes)),
        "mean_terminal_inventory": float(terminal_inventory.mean()),
        "mean_total_pnl": float(total_pnl.mean()),
        "mean_total_temporary_cost": float(total_temporary_cost.mean()),
        "mean_total_transient_cost": float(total_transient_cost.mean()),
        "mean_total_risk_cost": float(total_risk_cost.mean()),
        "mean_total_terminal_penalty": float(total_terminal_penalty.mean()),
    }


def _call_policy(
    policy_fn: PolicyFn,
    obs: np.ndarray,
    cfg: GBMExecutionConfig,
    rng: np.random.Generator,
) -> float:
    try:
        return float(policy_fn(obs, cfg, rng))
    except TypeError:
        return float(policy_fn(obs, cfg))
