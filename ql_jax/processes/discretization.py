"""Discretization schemes for stochastic processes."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class EulerDiscretization:
    """Standard Euler-Maruyama discretization.

    x(t+dt) = x(t) + drift(t,x)*dt + diffusion(t,x)*dW
    """

    def evolve(self, process, t0, x0, dt, dw):
        return x0 + process.drift(t0, x0) * dt + process.diffusion(t0, x0) * dw


@dataclass(frozen=True)
class EndEulerDiscretization:
    """End-point Euler discretization.

    Evaluates drift/diffusion at the end-point (predictor-corrector style).
    """

    def evolve(self, process, t0, x0, dt, dw):
        # Predictor step
        x_pred = x0 + process.drift(t0, x0) * dt + process.diffusion(t0, x0) * dw
        # Corrector: use averaged drift/diffusion
        drift_avg = 0.5 * (process.drift(t0, x0) + process.drift(t0 + dt, x_pred))
        diff_avg = 0.5 * (process.diffusion(t0, x0) + process.diffusion(t0 + dt, x_pred))
        return x0 + drift_avg * dt + diff_avg * dw
