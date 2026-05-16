"""Run oracle Monte Carlo evaluation for baseline behavior policies."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optimal_execution_phibe.config import GBMExecutionConfig
from optimal_execution_phibe.evaluation import evaluate_policy_mc
from optimal_execution_phibe.policies import (
    back_loaded_policy,
    front_loaded_policy,
    price_sensitive_policy,
    random_feasible_policy,
    twap_policy,
)

N_EPISODES = 1000
SUMMARY_PATH = Path("output/oracle_policy_evaluation_dt_0p025_seed_123.txt")
POLICIES = [
    ("twap", twap_policy),
    ("front_loaded", front_loaded_policy),
    ("back_loaded", back_loaded_policy),
    ("price_sensitive", price_sensitive_policy),
    ("random_feasible", random_feasible_policy),
]


def main() -> None:
    """Evaluate baseline policies and print a table of MC estimates."""

    cfg = GBMExecutionConfig()
    rows = [
        {
            "policy": name,
            **evaluate_policy_mc(
                cfg=cfg,
                policy_fn=policy_fn,
                n_episodes=N_EPISODES,
                seed=cfg.seed,
            ),
        }
        for name, policy_fn in POLICIES
    ]
    summary = _format_table(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"wrote summary: {SUMMARY_PATH}")


def _format_table(rows: list[dict[str, float | str]]) -> str:
    headers = [
        "policy",
        "mean_return",
        "std_return",
        "stderr_return",
        "mean_terminal_inventory",
        "mean_total_pnl",
        "mean_total_temporary_cost",
        "mean_total_transient_cost",
        "mean_total_risk_cost",
        "mean_total_terminal_penalty",
    ]
    widths = {
        header: max(len(header), *(len(_format_cell(row[header])) for row in rows))
        for header in headers
    }
    lines = [
        f"oracle Monte Carlo policy evaluation (n_episodes={N_EPISODES})",
        " | ".join(header.ljust(widths[header]) for header in headers),
        "-+-".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(
            " | ".join(_format_cell(row[header]).rjust(widths[header]) for header in headers)
        )
    return "\n".join(lines)


def _format_cell(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
