"""Credit default swap analytics engine."""

import jax.numpy as jnp


def cds_risky_annuity(spread, payment_dates, day_fractions, discount_fn,
                       survival_fn):
    """CDS risky PV01 (present value of 1bp of spread).

    Parameters
    ----------
    spread : float – CDS spread
    payment_dates : array
    day_fractions : array
    discount_fn : callable(t) -> DF
    survival_fn : callable(t) -> survival probability

    Returns
    -------
    rpv01 : float
    """
    rpv01 = 0.0
    for i in range(len(payment_dates)):
        rpv01 += day_fractions[i] * discount_fn(payment_dates[i]) * survival_fn(payment_dates[i])
    return rpv01


def cds_protection_leg(recovery, payment_dates, discount_fn, survival_fn,
                        notional=1.0, n_integration=4):
    """CDS protection leg PV.

    PV = (1-R) * ∫ P(t) * (-dQ(t))

    Parameters
    ----------
    recovery : float
    payment_dates : array – period boundaries
    discount_fn : callable(t) -> DF
    survival_fn : callable(t) -> survival prob
    notional : float
    n_integration : int – sub-intervals per period

    Returns
    -------
    pv : float
    """
    lgd = (1.0 - recovery) * notional
    pv = 0.0

    dates_ext = jnp.concatenate([jnp.array([0.0]), payment_dates])
    for i in range(len(dates_ext) - 1):
        t0 = dates_ext[i]
        t1 = dates_ext[i + 1]
        dt = (t1 - t0) / n_integration

        for j in range(n_integration):
            t_mid = t0 + (j + 0.5) * dt
            # Default probability in this interval
            dQ = survival_fn(t0 + j * dt) - survival_fn(t0 + (j + 1) * dt)
            pv += lgd * discount_fn(t_mid) * dQ

    return pv


def cds_npv(spread, recovery, payment_dates, day_fractions, discount_fn,
             survival_fn, notional=1.0, is_protection_buyer=True):
    """CDS net present value.

    Parameters
    ----------
    spread : float – contractual spread
    recovery : float
    payment_dates : array
    day_fractions : array
    discount_fn : callable(t) -> DF
    survival_fn : callable(t) -> survival prob
    notional : float
    is_protection_buyer : bool

    Returns
    -------
    npv : float
    """
    prot = cds_protection_leg(recovery, payment_dates, discount_fn,
                                survival_fn, notional)
    prem = spread * notional * cds_risky_annuity(
        spread, payment_dates, day_fractions, discount_fn, survival_fn
    )

    if is_protection_buyer:
        return prot - prem
    else:
        return prem - prot


def cds_fair_spread(recovery, payment_dates, day_fractions, discount_fn,
                     survival_fn, notional=1.0):
    """Fair CDS spread (par spread).

    Parameters
    ----------
    Same as cds_npv minus spread.

    Returns
    -------
    fair_spread : float
    """
    prot = cds_protection_leg(recovery, payment_dates, discount_fn,
                                survival_fn, notional)
    rpv01 = cds_risky_annuity(0.01, payment_dates, day_fractions, discount_fn,
                                survival_fn)

    return prot / (notional * rpv01)


def hazard_rate_from_spread(spread, recovery):
    """Approximate constant hazard rate from CDS spread.

    λ ≈ spread / (1 - R)

    Parameters
    ----------
    spread : float
    recovery : float

    Returns
    -------
    hazard_rate : float
    """
    return spread / (1.0 - recovery)
