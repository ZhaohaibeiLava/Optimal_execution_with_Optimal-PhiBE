# Optimal Execution PhiBE Simulator

First-stage research simulator for a finite-horizon GBM optimal execution problem
with transient impact.

This package provides:

- A Gymnasium-like environment API without depending on Gymnasium.
- Behavior policies for synthetic data collection.
- Offline trajectory dataset generation utilities.
- Unit tests covering simulator dynamics and policy feasibility.

The environment uses known parameters internally. Downstream learning algorithms
should consume only generated trajectory tuples and should not access true
dynamics, generators, transition densities, or parameters.

## Quickstart

```bash
python -m pytest
python scripts/01_generate_synthetic_data.py
```

The data-generation script writes:

```text
data/synthetic/gbm_execution_dt_0p025_seed_123.npz
```
