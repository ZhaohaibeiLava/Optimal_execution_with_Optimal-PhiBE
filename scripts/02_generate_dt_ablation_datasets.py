"""Generate synthetic GBM execution datasets for a dt-ablation study."""

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

DT_VALUES = [1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0, 1.0 / 160.0]
N_EPISODES = 1000
OUTPUT_DIR = Path("data/synthetic")
POLICY_MIXTURE = [
    twap_policy,
    noisy_twap_policy,
    front_loaded_policy,
    back_loaded_policy,
    price_sensitive_policy,
    random_feasible_policy,
]


def main() -> None:
    """Generate all ablation datasets and print a compact summary table."""

    rows = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dt in DT_VALUES:
        cfg = GBMExecutionConfig(T=1.0, dt=dt)
        dataset = generate_dataset(
            cfg=cfg,
            n_episodes=N_EPISODES,
            policy_mixture=POLICY_MIXTURE,
            seed=cfg.seed,
        )
        _assert_dataset_is_valid(dataset, cfg)

        output_path = OUTPUT_DIR / f"gbm_execution_dt_{_format_dt(dt)}_seed_{cfg.seed}.npz"
        np.savez_compressed(output_path, **dataset)
        summary = _summarize_dataset(dataset, cfg)
        summary["path"] = str(output_path)
        rows.append(summary)
        print(f"saved: {output_path}")

    _print_summary_table(rows)


def _assert_dataset_is_valid(
    dataset: dict[str, np.ndarray],
    cfg: GBMExecutionConfig,
    tol: float = 1e-10,
) -> None:
    prices = np.exp(
        np.concatenate([dataset["obs"][:, 3], dataset["next_obs"][:, 3]], axis=0)
    )
    inventories = np.concatenate(
        [dataset["obs"][:, 1], dataset["next_obs"][:, 1]], axis=0
    )
    rewards = dataset["rewards"]
    terminal_inventory = _terminal_inventory(dataset)
    total_sold = cfg.Q - terminal_inventory

    assert np.all(prices > 0.0), "Generated prices must be positive."
    assert np.all(inventories >= -tol), "Generated inventories must be nonnegative."
    assert np.all(np.isfinite(rewards)), "Generated rewards must be finite."
    assert np.all(total_sold <= cfg.Q + tol), "Total sold cannot exceed initial inventory."


def _summarize_dataset(
    dataset: dict[str, np.ndarray],
    cfg: GBMExecutionConfig,
) -> dict[str, float]:
    episode_ids = dataset["episode_id"]
    unique_episode_ids = np.unique(episode_ids)
    episode_rewards = np.array(
        [dataset["rewards"][episode_ids == episode_id].sum() for episode_id in unique_episode_ids]
    )
    terminal_mask = dataset["terminated"] | dataset["truncated"]
    terminal_inventory = _terminal_inventory(dataset)
    total_sold = cfg.Q - terminal_inventory

    actions = dataset["actions"]
    obs = dataset["obs"]
    next_obs = dataset["next_obs"]
    s = np.exp(obs[:, 3])
    s_next = np.exp(next_obs[:, 3])
    temporary_cost = cfg.eta * actions**2 * cfg.dt
    transient_cost = obs[:, 2] * actions * cfg.dt
    risk_cost = cfg.lambda_risk * obs[:, 1] * s * cfg.dt
    pnl = obs[:, 1] * (s_next - s)

    return {
        "dt": cfg.dt,
        "n_steps_per_episode": int(round(cfg.T / cfg.dt)),
        "n_transitions": int(dataset["rewards"].shape[0]),
        "mean_total_reward": float(episode_rewards.mean()),
        "mean_terminal_inventory": float(terminal_inventory.mean()),
        "fraction_inventory_terminated": float(dataset["terminated"][terminal_mask].mean()),
        "mean_total_sold": float(total_sold.mean()),
        "mean_temporary_cost": float(_episode_sums(temporary_cost, episode_ids).mean()),
        "mean_transient_cost": float(_episode_sums(transient_cost, episode_ids).mean()),
        "mean_risk_cost": float(_episode_sums(risk_cost, episode_ids).mean()),
        "mean_pnl": float(_episode_sums(pnl, episode_ids).mean()),
    }


def _terminal_inventory(dataset: dict[str, np.ndarray]) -> np.ndarray:
    terminal_mask = dataset["terminated"] | dataset["truncated"]
    return dataset["next_obs"][terminal_mask, 1]


def _episode_sums(values: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    return np.array(
        [values[episode_ids == episode_id].sum() for episode_id in np.unique(episode_ids)]
    )


def _format_dt(dt: float) -> str:
    return f"{dt:.8g}".replace(".", "p")


def _print_summary_table(rows: list[dict[str, float]]) -> None:
    headers = [
        "dt",
        "n_steps_per_episode",
        "n_transitions",
        "mean_total_reward",
        "mean_terminal_inventory",
        "fraction_inventory_terminated",
        "mean_total_sold",
        "mean_temporary_cost",
        "mean_transient_cost",
        "mean_risk_cost",
        "mean_pnl",
    ]
    widths = {
        header: max(len(header), *(len(_format_cell(row[header])) for row in rows))
        for header in headers
    }
    print()
    print("dt-ablation summary")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(_format_cell(row[header]).rjust(widths[header]) for header in headers))


def _format_cell(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
