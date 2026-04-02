"""Heston FD operator – 2D operator for Heston stochastic vol PDE."""

import jax.numpy as jnp
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import TridiagonalOperator


@dataclass(frozen=True)
class HestonFdmOperator:
    """2D finite-difference operator for the Heston PDE.

    PDE:
      dV/dt + (r-q-v/2)dV/dx + κ(θ-v)dV/dv
      + v/2 d²V/dx² + σ²v/2 d²V/dv² + ρσv d²V/dxdv - rV = 0

    Uses operator splitting (Douglas/Hundsdorfer) to handle cross-term.

    Parameters
    ----------
    r : float – risk-free rate
    q : float – dividend yield
    kappa : float – mean reversion speed
    theta : float – long-run variance
    sigma_v : float – vol of vol
    rho : float – correlation
    x_grid : array – log-spot grid
    v_grid : array – variance grid
    """
    r: float
    q: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    x_grid: jnp.ndarray
    v_grid: jnp.ndarray

    def build_x_operator(self, j):
        """Build x-direction operator at variance grid point j.

        Parameters
        ----------
        j : int – index into v_grid

        Returns
        -------
        TridiagonalOperator – operator in x dimension
        """
        v_j = self.v_grid[j]
        alpha = 0.5 * v_j
        beta = self.r - self.q - alpha

        x = self.x_grid
        n = len(x)
        dx = jnp.diff(x)

        lower = jnp.zeros(n - 1)
        diag = jnp.zeros(n)
        upper = jnp.zeros(n - 1)

        # Interior points
        dx_m = dx[:-1]
        dx_p = dx[1:]
        h = 0.5 * (dx_m + dx_p)

        lower = lower.at[1:].set(alpha / (dx_m * h) - beta / (dx_m + dx_p))
        diag = diag.at[1:-1].set(-alpha / (dx_p * h) - alpha / (dx_m * h) - self.r)
        upper = upper.at[:-1].set(alpha / (dx_p * h) + beta / (dx_m + dx_p))

        return TridiagonalOperator(lower, diag, upper)

    def build_v_operator(self, i):
        """Build v-direction operator at log-spot grid point i.

        Parameters
        ----------
        i : int – index into x_grid

        Returns
        -------
        TridiagonalOperator – operator in v dimension
        """
        v = self.v_grid
        n = len(v)
        dv = jnp.diff(v)

        lower = jnp.zeros(n - 1)
        diag = jnp.zeros(n)
        upper = jnp.zeros(n - 1)

        # Interior
        dv_m = dv[:-1]
        dv_p = dv[1:]
        h = 0.5 * (dv_m + dv_p)

        # CIR drift + diffusion in variance
        v_int = v[1:-1]
        drift = self.kappa * (self.theta - v_int)
        diff = 0.5 * self.sigma_v**2 * v_int

        lower = lower.at[1:].set(diff / (dv_m * h) - drift / (dv_m + dv_p))
        diag = diag.at[1:-1].set(-diff / (dv_p * h) - diff / (dv_m * h))
        upper = upper.at[:-1].set(diff / (dv_p * h) + drift / (dv_m + dv_p))

        return TridiagonalOperator(lower, diag, upper)

    def apply_cross_term(self, V_2d):
        """Apply the mixed derivative term ρσv d²V/dxdv.

        Uses central differences in both directions.

        Parameters
        ----------
        V_2d : array(nx, nv) – current solution on 2D grid

        Returns
        -------
        array(nx, nv) – contribution of cross-derivative
        """
        nx = len(self.x_grid)
        nv = len(self.v_grid)
        dx = jnp.diff(self.x_grid)
        dv = jnp.diff(self.v_grid)

        result = jnp.zeros_like(V_2d)

        # Interior (i=1..nx-2, j=1..nv-2)
        for i in range(1, nx - 1):
            for j in range(1, nv - 1):
                v_j = self.v_grid[j]
                coeff = self.rho * self.sigma_v * v_j
                dxh = 0.5 * (dx[i - 1] + dx[i])
                dvh = 0.5 * (dv[j - 1] + dv[j])

                d2_xv = (V_2d[i + 1, j + 1] - V_2d[i + 1, j - 1]
                          - V_2d[i - 1, j + 1] + V_2d[i - 1, j - 1]) / (4.0 * dxh * dvh)
                result = result.at[i, j].set(coeff * d2_xv)

        return result


def heston_douglas_step(V_2d, dt, op, theta_fd=0.5):
    """Douglas-Rachford or Hundsdorfer-Verwer ADI step for 2D Heston.

    Simplified ADI: solve x-direction implicitly, then v-direction implicitly.

    Parameters
    ----------
    V_2d : array(nx, nv) – current V
    dt : float – time step
    op : HestonFdmOperator
    theta_fd : float – implicitness parameter

    Returns
    -------
    V_new : array(nx, nv) – V at next time step
    """
    nx, nv = V_2d.shape

    # Step 1: Explicit + cross-term
    cross = op.apply_cross_term(V_2d)

    # Step 2: x-sweep (for each j)
    Y1 = jnp.zeros_like(V_2d)
    for j in range(nv):
        Lx = op.build_x_operator(j)
        rhs = V_2d[:, j] + dt * Lx.apply(V_2d[:, j]) + dt * cross[:, j]

        # Implicit correction in x
        lhs = TridiagonalOperator(
            -theta_fd * dt * Lx.lower,
            jnp.ones(nx) - theta_fd * dt * Lx.diag,
            -theta_fd * dt * Lx.upper,
        )
        Y1 = Y1.at[:, j].set(lhs.solve(rhs))

    # Step 3: v-sweep (for each i)
    V_new = jnp.zeros_like(V_2d)
    for i in range(nx):
        Lv = op.build_v_operator(i)
        rhs = Y1[i, :] + theta_fd * dt * Lv.apply(Y1[i, :] - V_2d[i, :])

        lhs = TridiagonalOperator(
            -theta_fd * dt * Lv.lower,
            jnp.ones(nv) - theta_fd * dt * Lv.diag,
            -theta_fd * dt * Lv.upper,
        )
        V_new = V_new.at[i, :].set(lhs.solve(rhs))

    return V_new
