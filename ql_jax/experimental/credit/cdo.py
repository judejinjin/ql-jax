"""Collateralized Debt Obligation (CDO) instruments."""

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


@dataclass
class CdoTranche:
    """A single CDO tranche defined by attachment/detachment points."""
    attachment: float  # lower attachment (e.g. 0.03 for 3%)
    detachment: float  # upper detachment (e.g. 0.07 for 7%)
    notional: float = 1.0
    spread: float = 0.0  # running spread (bps as decimal)

    @property
    def width(self):
        return self.detachment - self.attachment


@dataclass
class SyntheticCdo:
    """Synthetic CDO — a portfolio of CDS with tranched credit risk.

    Parameters
    ----------
    portfolio_notional : total portfolio notional
    tranches : list of CdoTranche objects
    recovery_rate : assumed uniform recovery rate
    """
    portfolio_notional: float
    tranches: Sequence[CdoTranche]
    recovery_rate: float = 0.4

    def tranche_loss(self, portfolio_loss_fraction, tranche_idx):
        """Compute tranche loss given portfolio loss fraction."""
        t = self.tranches[tranche_idx]
        loss = jnp.clip(portfolio_loss_fraction - t.attachment, 0.0,
                        t.width) / t.width
        return loss * t.notional


def expected_tranche_loss(default_probs, recovery_rate, attachment, detachment,
                          correlation, n_scenarios=10000, seed=42):
    """Compute expected tranche loss using Gaussian copula (LHP approximation).

    Parameters
    ----------
    default_probs : (n_names,) array of cumulative default probabilities
    recovery_rate : uniform recovery rate
    attachment, detachment : tranche boundaries
    correlation : flat pairwise correlation
    n_scenarios : number of MC scenarios
    seed : RNG seed

    Returns
    -------
    expected_loss : expected loss fraction of tranche
    """
    import jax
    from jax.scipy.stats import norm
    key = jax.random.PRNGKey(seed)

    n_names = len(default_probs)
    dp = jnp.asarray(default_probs)
    lgd = 1.0 - recovery_rate
    thresholds = norm.ppf(dp)

    rho_sqrt = jnp.sqrt(correlation)
    idio_sqrt = jnp.sqrt(1.0 - correlation)

    # Simulate common factor
    key, k1, k2 = jax.random.split(key, 3)
    Z = jax.random.normal(k1, (n_scenarios,))  # common factor
    eps = jax.random.normal(k2, (n_scenarios, n_names))  # idiosyncratic

    # Latent variables
    X = rho_sqrt * Z[:, None] + idio_sqrt * eps
    defaults = (X < thresholds[None, :]).astype(jnp.float64)
    losses = defaults * lgd / n_names  # each name has equal weight
    portfolio_losses = losses.sum(axis=1)

    # Tranche loss
    width = detachment - attachment
    tranche_losses = jnp.clip(portfolio_losses - attachment, 0.0, width) / width
    return float(tranche_losses.mean())
