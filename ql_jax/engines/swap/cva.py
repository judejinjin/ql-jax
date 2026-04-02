"""CVA (Credit Valuation Adjustment) swap engine."""

from __future__ import annotations

import jax.numpy as jnp


def cva_swap_engine(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    survival_fn,
    recovery: float = 0.4,
) -> float:
    """Compute unilateral CVA for an interest rate swap.

    CVA = (1 - R) * sum_i [ DF(t_i) * EE(t_i) * (Q(t_{i-1}) - Q(t_i)) ]

    Parameters
    ----------
    swap_mtm_fn : callable(t) -> expected positive exposure at time t
    exposure_dates : array of exposure monitoring dates (year fracs)
    discount_fn : t -> discount factor
    survival_fn : t -> counterparty survival probability
    recovery : recovery rate

    Returns
    -------
    float : CVA (positive = charge to counterparty)
    """
    n = exposure_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), exposure_dates[:-1]])

    cva = 0.0
    for i in range(n):
        t = float(exposure_dates[i])
        tp = float(t_prev[i])
        df = discount_fn(t)
        ee = jnp.maximum(swap_mtm_fn(t), 0.0)  # Expected positive exposure
        dQ = survival_fn(tp) - survival_fn(t)
        cva = cva + df * ee * dQ

    return (1.0 - recovery) * cva


def dva_swap_engine(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    own_survival_fn,
    own_recovery: float = 0.4,
) -> float:
    """Compute DVA (Debit Valuation Adjustment).

    DVA = (1 - R_own) * sum_i [ DF(t_i) * ENE(t_i) * (Q_own(t_{i-1}) - Q_own(t_i)) ]

    Parameters
    ----------
    swap_mtm_fn : callable(t) -> swap MTM (can be negative)
    exposure_dates : monitoring dates
    discount_fn : discount curve
    own_survival_fn : own default survival probability
    own_recovery : own recovery rate
    """
    n = exposure_dates.shape[0]
    t_prev = jnp.concatenate([jnp.zeros(1), exposure_dates[:-1]])

    dva = 0.0
    for i in range(n):
        t = float(exposure_dates[i])
        tp = float(t_prev[i])
        df = discount_fn(t)
        ene = jnp.maximum(-swap_mtm_fn(t), 0.0)  # Expected negative exposure
        dQ = own_survival_fn(tp) - own_survival_fn(t)
        dva = dva + df * ene * dQ

    return (1.0 - own_recovery) * dva


def bilateral_cva(
    swap_mtm_fn,
    exposure_dates: jnp.ndarray,
    discount_fn,
    cpty_survival_fn,
    own_survival_fn,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
) -> float:
    """Bilateral CVA = CVA - DVA."""
    cva = cva_swap_engine(
        swap_mtm_fn, exposure_dates, discount_fn,
        cpty_survival_fn, cpty_recovery,
    )
    dva = dva_swap_engine(
        swap_mtm_fn, exposure_dates, discount_fn,
        own_survival_fn, own_recovery,
    )
    return cva - dva
