"""Risk statistics: VaR, Expected Shortfall (CVaR), Sharpe ratio, Sortino ratio.

All operate on P&L or return arrays. Differentiable via JAX.
"""

import jax.numpy as jnp


def value_at_risk(samples, level=0.95):
    """Value at Risk at the given confidence level.

    VaR(α) = -quantile(samples, 1 - α)
    For a loss distribution, this is the maximum expected loss at level α.
    """
    return -jnp.percentile(samples, (1.0 - level) * 100.0)


def expected_shortfall(samples, level=0.95):
    """Expected Shortfall (CVaR / Conditional VaR) at the given confidence level.

    ES(α) = -E[X | X <= -VaR(α)]
    """
    var = value_at_risk(samples, level)
    tail = samples[samples <= -var]
    return jnp.where(len(tail) > 0, -jnp.mean(tail), -var)


def sharpe_ratio(returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio (assuming daily returns, 252 trading days)."""
    excess = returns - risk_free_rate / 252.0
    return jnp.mean(excess) / jnp.std(excess, ddof=1) * jnp.sqrt(252.0)


def sortino_ratio(returns, risk_free_rate=0.0):
    """Sortino ratio using downside deviation."""
    excess = returns - risk_free_rate / 252.0
    downside = jnp.minimum(excess, 0.0)
    downside_std = jnp.sqrt(jnp.mean(downside ** 2))
    return jnp.where(downside_std > 0, jnp.mean(excess) / downside_std * jnp.sqrt(252.0), jnp.inf)


def max_drawdown(cumulative_returns):
    """Maximum drawdown from a cumulative return series."""
    running_max = jnp.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns - running_max
    return jnp.min(drawdown)
