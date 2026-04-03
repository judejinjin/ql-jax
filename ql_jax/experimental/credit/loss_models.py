"""Credit loss models for basket/CDO pricing."""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CreditBasket:
    """A basket of credit names with default probabilities."""
    default_probs: jnp.ndarray  # (n_names,) cumulative default probs
    recovery_rates: jnp.ndarray  # (n_names,) recovery rates
    notionals: jnp.ndarray  # (n_names,) notionals

    @property
    def n_names(self):
        return len(self.default_probs)

    @property
    def total_notional(self):
        return float(self.notionals.sum())


def binomial_loss_model(basket, correlation, n_buckets=100):
    """Binomial loss model for credit basket.

    Computes the loss distribution using a single-factor Gaussian copula
    with conditional independence (binomial tower).

    Parameters
    ----------
    basket : CreditBasket
    correlation : flat pairwise correlation
    n_buckets : number of integration points for common factor

    Returns
    -------
    loss_levels : (n_names+1,) possible portfolio loss levels
    loss_probs : (n_names+1,) probability of each loss level
    """
    n = basket.n_names
    dp = basket.default_probs
    lgd = 1.0 - basket.recovery_rates
    weights = basket.notionals / basket.total_notional

    rho = jnp.sqrt(jnp.clip(correlation, 0.0, 1.0))
    thresholds = norm.ppf(jnp.clip(dp, 1e-15, 1.0 - 1e-15))

    # Gauss-Hermite quadrature for common factor
    z_pts, z_wts = _gauss_hermite(n_buckets)
    z_pts = z_pts * jnp.sqrt(2.0)
    z_wts = z_wts / jnp.sqrt(jnp.pi)

    # Conditional default probs
    cond_dp = norm.cdf((thresholds[None, :] - rho * z_pts[:, None]) /
                       jnp.sqrt(1.0 - correlation + 1e-15))

    # Build loss distribution via recursive convolution
    loss_per_name = lgd * weights
    loss_levels = jnp.zeros(n + 1)
    for i in range(n):
        loss_levels = loss_levels.at[i + 1].set(
            loss_levels[i] + float(loss_per_name[i]))

    # Probability for each scenario
    def _conditional_dist(z_idx):
        cp = cond_dp[z_idx]
        # Start with no defaults
        probs = jnp.zeros(n + 1).at[0].set(1.0)
        for i in range(n):
            new_probs = probs * (1.0 - cp[i])
            shifted = jnp.roll(probs, 1) * cp[i]
            shifted = shifted.at[0].set(0.0)
            probs = new_probs + shifted
        return probs

    # Average over common factor
    all_probs = jax.vmap(_conditional_dist)(jnp.arange(n_buckets))
    loss_probs = (all_probs * z_wts[:, None]).sum(axis=0)

    return loss_levels, loss_probs


def gaussian_lhp_loss(portfolio_dp, recovery_rate, correlation,
                      attachment, detachment, n_points=200):
    """Gaussian Large Homogeneous Pool (LHP) model for tranche pricing.

    Assumes infinite homogeneous pool. Gives closed-form expected tranche loss.

    Parameters
    ----------
    portfolio_dp : cumulative portfolio default probability
    recovery_rate : uniform recovery rate
    correlation : flat pairwise correlation
    attachment, detachment : tranche boundaries (as fractions of notional)
    n_points : quadrature points

    Returns
    -------
    expected_loss : expected tranche loss as fraction of tranche notional
    """
    lgd = 1.0 - recovery_rate
    rho = jnp.sqrt(jnp.clip(correlation, 1e-10, 1.0 - 1e-10))
    threshold = norm.ppf(jnp.clip(portfolio_dp, 1e-15, 1.0 - 1e-15))

    # Integration over common factor using Gauss-Hermite
    z_pts, z_wts = _gauss_hermite(n_points)
    z_pts = z_pts * jnp.sqrt(2.0)
    z_wts = z_wts / jnp.sqrt(jnp.pi)

    # Conditional default prob
    cond_dp = norm.cdf((threshold - rho * z_pts) /
                       jnp.sqrt(1.0 - correlation + 1e-15))

    # Conditional portfolio loss (LHP: loss = cond_dp * lgd)
    cond_loss = cond_dp * lgd

    width = detachment - attachment
    tranche_loss = jnp.clip(cond_loss - attachment, 0.0, width)
    expected = (tranche_loss * z_wts).sum() / width

    return float(expected)


def base_correlation_loss(tranche_spreads, detachment_points,
                          portfolio_dp, recovery_rate, base_corrs):
    """Map market tranche spreads to base correlation framework.

    Parameters
    ----------
    tranche_spreads : market spreads for each tranche
    detachment_points : detachment points for each tranche
    portfolio_dp : portfolio default probability
    recovery_rate : recovery rate
    base_corrs : base correlation for each detachment point

    Returns
    -------
    expected_losses : expected loss for each detachment tranche
    """
    losses = []
    for det, corr in zip(detachment_points, base_corrs):
        el = gaussian_lhp_loss(portfolio_dp, recovery_rate, corr, 0.0, det)
        losses.append(el)
    return jnp.array(losses)


def _gauss_hermite(n):
    """Gauss-Hermite quadrature points and weights."""
    import numpy as np
    pts, wts = np.polynomial.hermite.hermgauss(n)
    return jnp.array(pts), jnp.array(wts)


@dataclass
class NthToDefault:
    """Nth-to-default swap specification."""
    basket: CreditBasket
    nth: int  # which default triggers payment (1-indexed)
    premium_rate: float  # premium rate
    maturity: float  # years

    def default_leg_value(self, correlation, discount_factor=1.0, n_buckets=50):
        """Value of the default payment leg."""
        loss_levels, loss_probs = binomial_loss_model(
            self.basket, correlation, n_buckets)
        # P(nth or more defaults) = sum of probs for >= nth defaults
        cum_prob = loss_probs[self.nth:].sum()
        avg_lgd = float((1.0 - self.basket.recovery_rates).mean())
        return float(cum_prob * avg_lgd * discount_factor)


@dataclass
class CorrelationStructure:
    """Simple correlation term structure."""
    times: jnp.ndarray
    correlations: jnp.ndarray

    def correlation(self, t):
        return float(jnp.interp(t, self.times, self.correlations))
