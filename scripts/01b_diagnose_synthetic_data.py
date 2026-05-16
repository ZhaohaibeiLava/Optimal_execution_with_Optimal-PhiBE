"""Print diagnostics for the baseline synthetic GBM execution dataset."""

from __future__ import annotations

import io
from pathlib import Path
import sys
from contextlib import redirect_stdout

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optimal_execution_phibe.config import GBMExecutionConfig

DATA_PATH = Path("data/synthetic/gbm_execution_dt_0p025_seed_123.npz")
OUTPUT_PATH = Path("output/synthetic_data_diagnostics_dt_0p025_seed_123.txt")


def main() -> None:
    """Load the baseline dataset and print trajectory diagnostics."""

    cfg = GBMExecutionConfig()
    with np.load(DATA_PATH) as loaded:
        dataset = {key: loaded[key] for key in loaded.files}

    report_buffer = io.StringIO()
    with redirect_stdout(report_buffer):
        print(f"loaded: {DATA_PATH}")
        _print_dataset_summary(dataset, cfg)
        _print_reward_decomposition(dataset)
        _print_action_diagnostics(dataset)
        _print_state_diagnostics(dataset)

    report = report_buffer.getvalue()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"\nwrote summary: {OUTPUT_PATH}")


def _print_dataset_summary(dataset: dict[str, np.ndarray], cfg: GBMExecutionConfig) -> None:
    episode_ids = dataset["episode_id"]
    terminal_mask = dataset["terminated"] | dataset["truncated"]
    terminal_inventory = dataset["next_obs"][terminal_mask, 1]
    total_sold = cfg.Q - terminal_inventory
    n_episodes = int(np.unique(episode_ids).shape[0])
    n_transitions = int(dataset["rewards"].shape[0])
    n_steps_per_episode = int(round(cfg.T / cfg.dt))

    print()
    print("dataset summary")
    print(f"number of episodes: {n_episodes}")
    print(f"number of transitions: {n_transitions}")
    print(f"max possible transitions: {n_episodes * n_steps_per_episode}")
    print(f"average steps per episode: {n_transitions / n_episodes:.6f}")
    print(f"fraction terminated by inventory: {dataset['terminated'][terminal_mask].mean():.6f}")
    print(f"fraction truncated by horizon: {dataset['truncated'][terminal_mask].mean():.6f}")
    print(f"mean terminal inventory: {terminal_inventory.mean():.6f}")
    print(f"max terminal inventory: {terminal_inventory.max():.6f}")
    print(f"mean total sold: {total_sold.mean():.6f}")
    print(f"min total sold: {total_sold.min():.6f}")
    print(f"max total sold: {total_sold.max():.6f}")


def _print_reward_decomposition(dataset: dict[str, np.ndarray]) -> None:
    episode_ids = dataset["episode_id"]
    reconstructed_reward = (
        dataset["pnl"]
        - dataset["transient_cost"]
        - dataset["temporary_cost"]
        - dataset["risk_cost"]
        - dataset["terminal_penalty"]
    )
    decomposition_error = dataset["rewards"] - reconstructed_reward

    print()
    print("reward decomposition")
    print(f"mean total episode reward: {_mean_episode_sum(dataset['rewards'], episode_ids):.6f}")
    print(f"mean total pnl: {_mean_episode_sum(dataset['pnl'], episode_ids):.6f}")
    print(
        "mean total transient cost: "
        f"{_mean_episode_sum(dataset['transient_cost'], episode_ids):.6f}"
    )
    print(
        "mean total temporary cost: "
        f"{_mean_episode_sum(dataset['temporary_cost'], episode_ids):.6f}"
    )
    print(f"mean total risk cost: {_mean_episode_sum(dataset['risk_cost'], episode_ids):.6f}")
    print(
        "mean total terminal penalty: "
        f"{_mean_episode_sum(dataset['terminal_penalty'], episode_ids):.6f}"
    )
    print(
        "check: reward = pnl - transient_cost - temporary_cost "
        "- risk_cost - terminal_penalty"
    )
    print(f"max absolute decomposition error: {np.max(np.abs(decomposition_error)):.12f}")


def _print_action_diagnostics(dataset: dict[str, np.ndarray]) -> None:
    actions = dataset["actions"]
    clipped_actions = dataset["clipped_action"]
    clipped = np.abs(actions - clipped_actions) > 1e-12

    print()
    print("action diagnostics")
    print(f"mean action: {actions.mean():.6f}")
    print(f"std action: {actions.std():.6f}")
    print(f"min action: {actions.min():.6f}")
    print(f"max action: {actions.max():.6f}")
    print(f"fraction of actions clipped: {clipped.mean():.6f}")
    print(f"mean feasible_action_max: {dataset['feasible_action_max'].mean():.6f}")


def _print_state_diagnostics(dataset: dict[str, np.ndarray]) -> None:
    q = np.concatenate([dataset["obs"][:, 1], dataset["next_obs"][:, 1]])
    y = np.concatenate([dataset["obs"][:, 2], dataset["next_obs"][:, 2]])
    log_s = np.concatenate([dataset["obs"][:, 3], dataset["next_obs"][:, 3]])
    prices = np.concatenate([dataset["S"], dataset["S_next"]])

    print()
    print("state diagnostics")
    print(f"min q: {q.min():.12f}")
    print(f"max q: {q.max():.12f}")
    print(f"min y: {y.min():.12f}")
    print(f"max y: {y.max():.12f}")
    print(f"min S: {prices.min():.12f}")
    print(f"max S: {prices.max():.12f}")
    print(f"min logS: {log_s.min():.12f}")
    print(f"max logS: {log_s.max():.12f}")
    print(f"check all q >= -1e-10: {bool(np.all(q >= -1e-10))}")
    print(f"check all S > 0: {bool(np.all(prices > 0.0))}")
    print(f"check all rewards finite: {bool(np.all(np.isfinite(dataset['rewards'])))}")


def _mean_episode_sum(values: np.ndarray, episode_ids: np.ndarray) -> float:
    episode_sums = np.array(
        [values[episode_ids == episode_id].sum() for episode_id in np.unique(episode_ids)]
    )
    return float(episode_sums.mean())


if __name__ == "__main__":
    main()
