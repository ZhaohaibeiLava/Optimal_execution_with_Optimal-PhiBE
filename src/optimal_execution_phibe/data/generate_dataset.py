"""Offline dataset generation for GBM execution trajectories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.envs import GBMExecutionEnv

PolicyFn = Callable[..., float]


def rollout_episode(
    env: GBMExecutionEnv,
    policy_fn: PolicyFn,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Roll out one episode using a behavior policy."""

    obs, _ = env.reset(seed=int(rng.integers(0, np.iinfo(np.uint32).max)))
    rows: dict[str, list[Any]] = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "terminated": [],
        "truncated": [],
        "step_id": [],
    }

    done = False
    while not done:
        action = _call_policy(policy_fn, obs, env.cfg, rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rows["obs"].append(obs)
        rows["actions"].append(float(action))
        rows["rewards"].append(float(reward))
        rows["next_obs"].append(next_obs)
        rows["terminated"].append(bool(terminated))
        rows["truncated"].append(bool(truncated))
        rows["step_id"].append(env.step_count - 1)

        obs = next_obs
        done = terminated or truncated

    return {
        "obs": np.asarray(rows["obs"], dtype=np.float64),
        "actions": np.asarray(rows["actions"], dtype=np.float64),
        "rewards": np.asarray(rows["rewards"], dtype=np.float64),
        "next_obs": np.asarray(rows["next_obs"], dtype=np.float64),
        "terminated": np.asarray(rows["terminated"], dtype=bool),
        "truncated": np.asarray(rows["truncated"], dtype=bool),
        "step_id": np.asarray(rows["step_id"], dtype=np.int64),
    }


def generate_dataset(
    cfg: GBMExecutionConfig,
    n_episodes: int,
    policy_mixture: Sequence[PolicyFn] | dict[PolicyFn, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate a transition dataset from a mixture of behavior policies."""

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive.")
    rng = np.random.default_rng(seed)
    policies, probs = _normalize_policy_mixture(policy_mixture)

    episodes: list[dict[str, np.ndarray]] = []
    episode_ids: list[np.ndarray] = []
    for episode_id in range(n_episodes):
        policy_idx = int(rng.choice(len(policies), p=probs))
        env = GBMExecutionEnv(cfg)
        episode = rollout_episode(env, policies[policy_idx], rng)
        episodes.append(episode)
        episode_ids.append(
            np.full(episode["rewards"].shape[0], episode_id, dtype=np.int64)
        )

    dataset = {
        "obs": np.concatenate([ep["obs"] for ep in episodes], axis=0),
        "actions": np.concatenate([ep["actions"] for ep in episodes], axis=0),
        "rewards": np.concatenate([ep["rewards"] for ep in episodes], axis=0),
        "next_obs": np.concatenate([ep["next_obs"] for ep in episodes], axis=0),
        "terminated": np.concatenate([ep["terminated"] for ep in episodes], axis=0),
        "truncated": np.concatenate([ep["truncated"] for ep in episodes], axis=0),
        "episode_id": np.concatenate(episode_ids, axis=0),
        "step_id": np.concatenate([ep["step_id"] for ep in episodes], axis=0),
    }
    return dataset


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


def _normalize_policy_mixture(
    policy_mixture: Sequence[PolicyFn] | dict[PolicyFn, float],
) -> tuple[list[PolicyFn], np.ndarray | None]:
    if isinstance(policy_mixture, dict):
        if not policy_mixture:
            raise ValueError("policy_mixture must not be empty.")
        policies = list(policy_mixture.keys())
        weights = np.asarray(list(policy_mixture.values()), dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("Policy mixture weights must be nonnegative with positive sum.")
        return policies, weights / weights.sum()

    policies = list(policy_mixture)
    if not policies:
        raise ValueError("policy_mixture must not be empty.")
    return policies, None
