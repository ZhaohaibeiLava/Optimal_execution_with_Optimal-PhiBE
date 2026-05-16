"""Generate a small synthetic GBM execution dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.data import generate_dataset
from optimal_execution_phibe.policies import (
    back_loaded_policy,
    front_loaded_policy,
    noisy_twap_policy,
    price_sensitive_policy,
    random_feasible_policy,
    twap_policy,
)


def main() -> None:
    cfg = GBMExecutionConfig()
    dataset = generate_dataset(
        cfg=cfg,
        n_episodes=100,
        policy_mixture=[
            twap_policy,
            noisy_twap_policy,
            front_loaded_policy,
            back_loaded_policy,
            price_sensitive_policy,
            random_feasible_policy,
        ],
        seed=cfg.seed,
    )

    output_path = Path("data/synthetic/gbm_execution_dt_0p025_seed_123.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)

    terminal_mask = dataset["terminated"] | dataset["truncated"]
    terminal_next_obs = dataset["next_obs"][terminal_mask]
    terminal_inventory = terminal_next_obs[:, 1]
    episode_ids = dataset["episode_id"]
    episode_rewards = np.array(
        [dataset["rewards"][episode_ids == i].sum() for i in np.unique(episode_ids)]
    )
    initial_inventory = cfg.Q
    total_sold = initial_inventory - terminal_inventory

    print(f"saved: {output_path}")
    print(f"number of transitions: {dataset['rewards'].shape[0]}")
    print(f"mean reward: {dataset['rewards'].mean():.6f}")
    print(f"mean terminal inventory: {terminal_inventory.mean():.6f}")
    print(f"mean total episode reward: {episode_rewards.mean():.6f}")
    print(f"mean turnover / total sold: {total_sold.mean():.6f}")
    print(f"fraction terminated by inventory: {dataset['terminated'][terminal_mask].mean():.6f}")


if __name__ == "__main__":
    main()
