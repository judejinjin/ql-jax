"""Heston FD operator – vectorized 2D ADI for Heston stochastic vol PDE."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import tridiag_solve


@dataclass(frozen=True)
class HestonFdmOperator:
    """Pre-built 2D finite-difference operators for the Heston PDE.

    Stores all operator tridiagonals as batched arrays for vectorized ADI.

    Attributes
    ----------
    x_lower, x_diag, x_upper : (nv, nx) or (nv, nx-1) – x-operators per v-slice
    v_lower, v_diag, v_upper : (nv,) or (nv-1,) – single v-operator (same for all i)
    cross_coeff : (nx, nv) – cross-derivative coefficient grid
    cross_dxh : (nx,) – half-widths in x for interior points
    cross_dvh : (nv,) – half-widths in v for interior points
    """
    # X-direction operators: shape (nv, nx) / (nv, nx-1)
    x_lower: jnp.ndarray
    x_diag: jnp.ndarray
    x_upper: jnp.ndarray
    # V-direction operator (same for all x): shape (nv-1,) / (nv,) / (nv-1,)
    v_lower: jnp.ndarray
    v_diag: jnp.ndarray
    v_upper: jnp.ndarray
    # Cross-term data
    cross_coeff: jnp.ndarray  # (nx, nv) – ρσv_j coefficient
    cross_dxh: jnp.ndarray    # (nx-2,) – half-widths in x (interior)
    cross_dvh: jnp.ndarray    # (nv-2,) – half-widths in v (interior)


def build_heston_operator(r, q, kappa, theta, sigma_v, rho, x_grid, v_grid):
    """Build all Heston FD operator data in vectorized form.

    Parameters
    ----------
    r, q : float – risk-free rate, dividend yield
    kappa, theta, sigma_v, rho : float – Heston params
    x_grid : (nx,) – log-spot grid
    v_grid : (nv,) – variance grid

    Returns
    -------
    HestonFdmOperator
    """
    nx = len(x_grid)
    nv = len(v_grid)
    dx = jnp.diff(x_grid)
    dv = jnp.diff(v_grid)

    # ── X-direction operators, one per v-slice ──
    dx_m = dx[:-1]  # (nx-2,)
    dx_p = dx[1:]   # (nx-2,)
    hx = 0.5 * (dx_m + dx_p)  # (nx-2,)

    # Broadcast: alpha(j) = 0.5*v_j, beta(j) = r-q-0.5*v_j
    alpha = 0.5 * v_grid  # (nv,)
    beta = r - q - alpha  # (nv,)

    # Interior coefficients: shape (nv, nx-2)
    # lower[j, :] = alpha_j / (dx_m * hx) - beta_j / (dx_m + dx_p)
    x_lower_int = alpha[:, None] / (dx_m[None, :] * hx[None, :]) - beta[:, None] / ((dx_m + dx_p)[None, :])
    x_diag_int = -alpha[:, None] / (dx_p[None, :] * hx[None, :]) - alpha[:, None] / (dx_m[None, :] * hx[None, :]) - r
    x_upper_int = alpha[:, None] / (dx_p[None, :] * hx[None, :]) + beta[:, None] / ((dx_m + dx_p)[None, :])

    # Pad to full size
    # In _tridiag_apply: lower[k] -> M[k+1,k], upper[k] -> M[k,k+1]
    # Interior lower M[1,0]..M[nx-2,nx-3] -> lower[0]..lower[nx-3], pad zero at end
    x_lower_full = jnp.zeros((nv, nx - 1))
    x_lower_full = x_lower_full.at[:, :-1].set(x_lower_int)
    x_diag_full = jnp.zeros((nv, nx))
    x_diag_full = x_diag_full.at[:, 1:-1].set(x_diag_int)
    # Interior upper M[1,2]..M[nx-2,nx-1] -> upper[1]..upper[nx-2], pad zero at start
    x_upper_full = jnp.zeros((nv, nx - 1))
    x_upper_full = x_upper_full.at[:, 1:].set(x_upper_int)

    # ── V-direction operator (same for all x) ──
    dv_m = dv[:-1]  # (nv-2,)
    dv_p = dv[1:]   # (nv-2,)
    hv = 0.5 * (dv_m + dv_p)
    v_int = v_grid[1:-1]
    drift_v = kappa * (theta - v_int)
    diff_v = 0.5 * sigma_v**2 * v_int

    v_lower_full = jnp.zeros(nv - 1)
    v_lower_full = v_lower_full.at[:-1].set(diff_v / (dv_m * hv) - drift_v / (dv_m + dv_p))
    v_diag_full = jnp.zeros(nv)
    v_diag_full = v_diag_full.at[1:-1].set(-diff_v / (dv_p * hv) - diff_v / (dv_m * hv))
    v_upper_full = jnp.zeros(nv - 1)
    v_upper_full = v_upper_full.at[1:].set(diff_v / (dv_p * hv) + drift_v / (dv_m + dv_p))

    # ── Cross-derivative data ──
    cross_coeff = jnp.outer(jnp.ones(nx), rho * sigma_v * v_grid)  # (nx, nv)
    cross_dxh = 0.5 * (dx[:-1] + dx[1:])   # (nx-2,)
    cross_dvh = 0.5 * (dv[:-1] + dv[1:])   # (nv-2,)

    return HestonFdmOperator(
        x_lower=x_lower_full, x_diag=x_diag_full, x_upper=x_upper_full,
        v_lower=v_lower_full, v_diag=v_diag_full, v_upper=v_upper_full,
        cross_coeff=cross_coeff, cross_dxh=cross_dxh, cross_dvh=cross_dvh,
    )


def _apply_cross_term(V_2d, op):
    """Vectorized cross-derivative ρσv d²V/dxdv."""
    # Central differences on interior (1:-1, 1:-1)
    d2 = (V_2d[2:, 2:] - V_2d[2:, :-2] - V_2d[:-2, 2:] + V_2d[:-2, :-2])
    denom = 4.0 * op.cross_dxh[:, None] * op.cross_dvh[None, :]
    interior = op.cross_coeff[1:-1, 1:-1] * d2 / denom
    result = jnp.zeros_like(V_2d)
    result = result.at[1:-1, 1:-1].set(interior)
    return result


def _tridiag_apply(lower, diag, upper, v):
    """Apply tridiagonal matrix to vector."""
    result = diag * v
    result = result.at[1:].add(lower * v[:-1])
    result = result.at[:-1].add(upper * v[1:])
    return result


def _x_sweep_single(carry, j_data):
    """Single x-direction tridiagonal solve (for vmap over v-slices)."""
    # j_data = (x_lower_j, x_diag_j, x_upper_j, rhs_j)
    x_lo, x_di, x_up, v_col, cross_col, dt, theta_fd = j_data
    nx = v_col.shape[0]
    # Explicit: rhs = V + dt * Lx(V) + dt * cross
    Lx_v = _tridiag_apply(x_lo, x_di, x_up, v_col)
    rhs = v_col + dt * Lx_v + dt * cross_col
    # Implicit solve
    sol = tridiag_solve(-theta_fd * dt * x_lo,
                        jnp.ones(nx) - theta_fd * dt * x_di,
                        -theta_fd * dt * x_up, rhs)
    return None, sol


def _v_sweep_single(v_lo, v_di, v_up, y1_row, v0_row, dt, theta_fd):
    """Single v-direction tridiagonal solve."""
    nv = y1_row.shape[0]
    Lv_delta = _tridiag_apply(v_lo, v_di, v_up, y1_row - v0_row)
    rhs = y1_row + theta_fd * dt * Lv_delta
    return tridiag_solve(-theta_fd * dt * v_lo,
                         jnp.ones(nv) - theta_fd * dt * v_di,
                         -theta_fd * dt * v_up, rhs)


def heston_douglas_step(V_2d, dt, op, theta_fd=0.5):
    """Douglas ADI step for 2D Heston.

    Step 1: Y0 = V + dt*(Lx + Lv + Lc)(V)       [full explicit]
    Step 2: (I - θ dt Lx) Y1 = Y0 - θ dt Lx(V)  [implicit x correction]
    Step 3: (I - θ dt Lv) V_new = Y1 - θ dt Lv(V) [implicit v correction]
    """
    nx, nv = V_2d.shape

    # Apply operators to V
    def apply_Lx(x_lo, x_di, x_up, col):
        return _tridiag_apply(x_lo, x_di, x_up, col)
    def apply_Lv(row):
        return _tridiag_apply(op.v_lower, op.v_diag, op.v_upper, row)

    LxV_T = jax.vmap(apply_Lx)(op.x_lower, op.x_diag, op.x_upper, V_2d.T)
    LxV = LxV_T.T
    LvV = jax.vmap(apply_Lv)(V_2d)
    LcV = _apply_cross_term(V_2d, op)

    # Step 1: full explicit predictor
    Y0 = V_2d + dt * (LxV + LvV + LcV)

    # Step 2: implicit x-correction (solve per v-slice, i.e. per column)
    def x_solve(x_lo, x_di, x_up, y0_col, lxv_col):
        rhs = y0_col - theta_fd * dt * lxv_col
        return tridiag_solve(-theta_fd * dt * x_lo,
                             jnp.ones(nx) - theta_fd * dt * x_di,
                             -theta_fd * dt * x_up, rhs)

    Y1_T = jax.vmap(x_solve)(op.x_lower, op.x_diag, op.x_upper, Y0.T, LxV.T)
    Y1 = Y1_T.T

    # Step 3: implicit v-correction (solve per x-row)
    def v_solve(y1_row, lvv_row):
        rhs = y1_row - theta_fd * dt * lvv_row
        return tridiag_solve(-theta_fd * dt * op.v_lower,
                             jnp.ones(nv) - theta_fd * dt * op.v_diag,
                             -theta_fd * dt * op.v_upper, rhs)

    V_new = jax.vmap(v_solve)(Y1, LvV)
    return V_new
