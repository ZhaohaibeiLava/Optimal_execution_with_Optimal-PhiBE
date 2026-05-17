"""Generate synthetic GBM execution datasets for all baseline dt values."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

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
DATA_DIR = Path("data/synthetic")
SUMMARY_DIR = Path("output")
POLICY_MIXTURE = [
    twap_policy,
    noisy_twap_policy,
    front_loaded_policy,
    back_loaded_policy,
    price_sensitive_policy,
    random_feasible_policy,
]


def main() -> None:
    """Generate all synthetic datasets and write per-dt summaries."""

    rows = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for dt in DT_VALUES:
        cfg = GBMExecutionConfig(T=1.0, dt=dt)
        n_steps = int(round(cfg.T / cfg.dt))
        dataset = generate_dataset(
            cfg=cfg,
            n_episodes=N_EPISODES,
            policy_mixture=POLICY_MIXTURE,
            seed=cfg.seed,
        )
        _assert_dataset_is_valid(dataset, cfg)

        dt_label = _format_dt(dt)
        data_path = DATA_DIR / f"gbm_execution_dt_{dt_label}_seed_{cfg.seed}.npz"
        summary_path = (
            SUMMARY_DIR
            / f"synthetic_data_generation_summary_N_{n_steps}_dt_{dt_label}_seed_{cfg.seed}.txt"
        )
        np.savez_compressed(data_path, **dataset)

        summary = _summarize_dataset(dataset, cfg)
        summary["data_path"] = str(data_path)
        summary["summary_path"] = str(summary_path)
        rows.append(summary)

        per_dt_summary = _format_single_summary(summary)
        summary_path.write_text(per_dt_summary + "\n", encoding="utf-8")
        print(per_dt_summary)
        print(f"wrote summary: {summary_path}")
        print()

    table = _format_summary_table(rows)
    table_path = SUMMARY_DIR / f"synthetic_data_generation_dt_ablation_seed_{GBMExecutionConfig().seed}.txt"
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"wrote summary: {table_path}")


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
) -> dict[str, Any]:
    episode_ids = dataset["episode_id"]
    unique_episode_ids = np.unique(episode_ids)
    episode_rewards = np.array(
        [dataset["rewards"][episode_ids == episode_id].sum() for episode_id in unique_episode_ids]
    )
    terminal_mask = dataset["terminated"] | dataset["truncated"]
    terminal_inventory = _terminal_inventory(dataset)
    total_sold = cfg.Q - terminal_inventory

    return {
        "dt": cfg.dt,
        "n_steps_per_episode": int(round(cfg.T / cfg.dt)),
        "n_episodes": int(unique_episode_ids.shape[0]),
        "n_transitions": int(dataset["rewards"].shape[0]),
        "max_possible_transitions": int(unique_episode_ids.shape[0] * round(cfg.T / cfg.dt)),
        "average_steps_per_episode": float(dataset["rewards"].shape[0] / unique_episode_ids.shape[0]),
        "mean_reward": float(dataset["rewards"].mean()),
        "mean_total_reward": float(episode_rewards.mean()),
        "mean_terminal_inventory": float(terminal_inventory.mean()),
        "max_terminal_inventory": float(terminal_inventory.max()),
        "fraction_inventory_terminated": float(dataset["terminated"][terminal_mask].mean()),
        "fraction_horizon_truncated": float(dataset["truncated"][terminal_mask].mean()),
        "mean_total_sold": float(total_sold.mean()),
        "min_total_sold": float(total_sold.min()),
        "max_total_sold": float(total_sold.max()),
        "mean_total_pnl": float(_episode_sums(dataset["pnl"], episode_ids).mean()),
        "mean_total_temporary_cost": float(
            _episode_sums(dataset["temporary_cost"], episode_ids).mean()
        ),
        "mean_total_transient_cost": float(
            _episode_sums(dataset["transient_cost"], episode_ids).mean()
        ),
        "mean_total_risk_cost": float(_episode_sums(dataset["risk_cost"], episode_ids).mean()),
        "mean_total_terminal_penalty": float(
            _episode_sums(dataset["terminal_penalty"], episode_ids).mean()
        ),
        "per_policy": _summarize_by_policy(dataset, cfg),
    }


def _summarize_by_policy(
    dataset: dict[str, np.ndarray],
    cfg: GBMExecutionConfig,
) -> list[dict[str, float | int | str]]:
    episode_ids = dataset["episode_id"]
    policy_rows: list[dict[str, float | int | str]] = []
    terminal_mask = dataset["terminated"] | dataset["truncated"]

    for policy_id in np.unique(dataset["policy_id"]):
        policy_mask = dataset["policy_id"] == policy_id
        policy_episode_ids = np.unique(episode_ids[policy_mask])
        policy_terminal_mask = terminal_mask & policy_mask
        terminal_inventory = dataset["next_obs"][policy_terminal_mask, 1]
        total_sold = cfg.Q - terminal_inventory
        policy_name = str(dataset["policy_name"][policy_mask][0])

        policy_rows.append(
            {
                "policy_id": int(policy_id),
                "policy_name": policy_name,
                "n_episodes": int(policy_episode_ids.shape[0]),
                "n_transitions": int(policy_mask.sum()),
                "mean_total_reward": float(
                    _episode_sums(dataset["rewards"], episode_ids, policy_episode_ids).mean()
                ),
                "mean_terminal_inventory": float(terminal_inventory.mean()),
                "mean_total_sold": float(total_sold.mean()),
                "mean_total_pnl": float(
                    _episode_sums(dataset["pnl"], episode_ids, policy_episode_ids).mean()
                ),
                "mean_total_temporary_cost": float(
                    _episode_sums(
                        dataset["temporary_cost"], episode_ids, policy_episode_ids
                    ).mean()
                ),
                "mean_total_transient_cost": float(
                    _episode_sums(
                        dataset["transient_cost"], episode_ids, policy_episode_ids
                    ).mean()
                ),
                "mean_total_risk_cost": float(
                    _episode_sums(dataset["risk_cost"], episode_ids, policy_episode_ids).mean()
                ),
                "mean_total_terminal_penalty": float(
                    _episode_sums(
                        dataset["terminal_penalty"], episode_ids, policy_episode_ids
                    ).mean()
                ),
            }
        )

    return policy_rows


def _terminal_inventory(dataset: dict[str, np.ndarray]) -> np.ndarray:
    terminal_mask = dataset["terminated"] | dataset["truncated"]
    return dataset["next_obs"][terminal_mask, 1]


def _episode_sums(
    values: np.ndarray,
    episode_ids: np.ndarray,
    selected_episode_ids: np.ndarray | None = None,
) -> np.ndarray:
    ids = np.unique(episode_ids) if selected_episode_ids is None else selected_episode_ids
    return np.array(
        [values[episode_ids == episode_id].sum() for episode_id in ids],
        dtype=np.float64,
    )


def _format_single_summary(summary: dict[str, Any]) -> str:
    keys = [
        "data_path",
        "dt",
        "n_steps_per_episode",
        "n_episodes",
        "n_transitions",
        "max_possible_transitions",
        "average_steps_per_episode",
        "mean_reward",
        "mean_total_reward",
        "mean_terminal_inventory",
        "max_terminal_inventory",
        "fraction_inventory_terminated",
        "fraction_horizon_truncated",
        "mean_total_sold",
        "min_total_sold",
        "max_total_sold",
        "mean_total_pnl",
        "mean_total_temporary_cost",
        "mean_total_transient_cost",
        "mean_total_risk_cost",
        "mean_total_terminal_penalty",
    ]
    lines = ["synthetic data generation summary"]
    for key in keys:
        lines.append(f"{key}: {_format_value(summary[key])}")
    lines.append("")
    lines.append(_format_policy_table(summary["per_policy"]))
    return "\n".join(lines)


def _format_policy_table(rows: list[dict[str, float | int | str]]) -> str:
    headers = [
        "policy_id",
        "policy_name",
        "n_episodes",
        "n_transitions",
        "mean_total_reward",
        "mean_terminal_inventory",
        "mean_total_sold",
        "mean_total_pnl",
        "mean_total_temporary_cost",
        "mean_total_transient_cost",
        "mean_total_risk_cost",
        "mean_total_terminal_penalty",
    ]
    widths = {
        header: max(len(header), *(len(_format_value(row[header])) for row in rows))
        for header in headers
    }
    lines = ["per-policy summary"]
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        lines.append(
            " | ".join(_format_value(row[header]).rjust(widths[header]) for header in headers)
        )
    return "\n".join(lines)


def _format_summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "dt",
        "n_steps_per_episode",
        "n_transitions",
        "mean_total_reward",
        "mean_terminal_inventory",
        "fraction_inventory_terminated",
        "fraction_horizon_truncated",
        "mean_total_sold",
        "mean_total_pnl",
        "mean_total_temporary_cost",
        "mean_total_transient_cost",
        "mean_total_risk_cost",
        "mean_total_terminal_penalty",
    ]
    widths = {
        header: max(len(header), *(len(_format_value(row[header])) for row in rows))
        for header in headers
    }
    lines = ["dt-ablation synthetic data generation summary"]
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        lines.append(
            " | ".join(_format_value(row[header]).rjust(widths[header]) for header in headers)
        )
    return "\n".join(lines)


def _format_dt(dt: float) -> str:
    return f"{dt:.8g}".replace(".", "p")


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        return f"{float(value):.6f}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
