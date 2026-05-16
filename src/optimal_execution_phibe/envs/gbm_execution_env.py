"""GBM optimal execution environment with transient impact."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimal_execution_phibe.config import GBMExecutionConfig


class GBMExecutionEnv:
    """Finite-horizon GBM execution simulator with a Gymnasium-like API."""

    def __init__(self, cfg: GBMExecutionConfig):
        self.cfg = cfg
        self.N = int(round(cfg.T / cfg.dt))
        self.rng = np.random.default_rng(cfg.seed)
        self._obs = np.array([0.0, cfg.Q, cfg.y0, np.log(cfg.S0)], dtype=np.float64)
        self.step_count = 0

    @property
    def observation(self) -> np.ndarray:
        """Return a copy of the current observation."""

        return self._obs.copy()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode and optionally reseed the environment RNG."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._obs = np.array(
            [0.0, self.cfg.Q, self.cfg.y0, np.log(self.cfg.S0)],
            dtype=np.float64,
        )
        return self.observation, {"step_count": self.step_count}

    def step(
        self, action: float | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one simulation step using a constant selling rate."""

        raw_action = self._coerce_action(action)
        t, q, y, log_s = self._obs
        feasible_action_max = min(self.cfg.u_bar, q / self.cfg.dt)
        clipped_action = float(np.clip(raw_action, 0.0, feasible_action_max))

        s = float(np.exp(log_s))
        eps = float(self.rng.normal())
        log_s_next = float(
            log_s
            + (self.cfg.mu - 0.5 * self.cfg.sigma**2) * self.cfg.dt
            + self.cfg.sigma * np.sqrt(self.cfg.dt) * eps
        )
        s_next = float(np.exp(log_s_next))
        q_next = float(max(q - clipped_action * self.cfg.dt, 0.0))

        decay = float(np.exp(-self.cfg.rho * self.cfg.dt))
        y_next = float(
            decay * y
            + (self.cfg.kappa / self.cfg.rho) * (1.0 - decay) * clipped_action
        )

        pnl = float(q * (s_next - s))
        transient_cost = float(y * clipped_action * self.cfg.dt)
        temporary_cost = float(self.cfg.eta * clipped_action**2 * self.cfg.dt)
        risk_cost = float(self.cfg.lambda_risk * q * s * self.cfg.dt)
        terminal_penalty = 0.0
        reward = pnl - transient_cost - temporary_cost - risk_cost

        self.step_count += 1
        t_next = min(self.step_count * self.cfg.dt, self.cfg.T)
        terminated = q_next <= self.cfg.inventory_tol
        truncated = self.step_count >= self.N
        if truncated:
            terminal_penalty = float(self.cfg.chi * q_next**2)
            reward -= terminal_penalty

        self._obs = np.array([t_next, q_next, y_next, log_s_next], dtype=np.float64)
        info: dict[str, Any] = {
            "raw_action": raw_action,
            "clipped_action": clipped_action,
            "feasible_action_max": float(feasible_action_max),
            "S": s,
            "S_next": s_next,
            "pnl": pnl,
            "transient_cost": transient_cost,
            "temporary_cost": temporary_cost,
            "risk_cost": risk_cost,
            "terminal_penalty": terminal_penalty,
            "q_next": q_next,
            "y_next": y_next,
            "logS_next": log_s_next,
            "step_count": self.step_count,
        }
        return self.observation, float(reward), bool(terminated), bool(truncated), info

    @staticmethod
    def _coerce_action(action: float | np.ndarray) -> float:
        arr = np.asarray(action, dtype=np.float64)
        if arr.shape == ():
            return float(arr)
        if arr.size != 1:
            raise ValueError("Action must be a scalar or one-element array.")
        return float(arr.reshape(-1)[0])
