"""Configuration objects for optimal execution simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GBMExecutionConfig:
    """Parameters for the finite-horizon GBM execution environment."""

    T: float = 1.0
    dt: float = 1.0 / 40.0
    Q: float = 1.0
    S0: float = 1.0
    y0: float = 0.0
    mu: float = 0.0
    sigma: float = 0.2
    rho: float = 5.0
    kappa: float = 0.1
    eta: float = 0.01
    lambda_risk: float = 0.01
    chi: float = 50.0
    u_bar: float = 5.0
    inventory_tol: float = 1e-8
    seed: int = 123

    def __post_init__(self) -> None:
        """Validate parameter ranges."""

        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.Q <= 0:
            raise ValueError("Q must be positive.")
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")
        if self.sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if self.rho <= 0:
            raise ValueError("rho must be positive.")
        if self.kappa < 0:
            raise ValueError("kappa must be nonnegative.")
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.lambda_risk < 0:
            raise ValueError("lambda_risk must be nonnegative.")
        if self.chi < 0:
            raise ValueError("chi must be nonnegative.")
        if self.u_bar <= 0:
            raise ValueError("u_bar must be positive.")
