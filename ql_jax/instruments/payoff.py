"""Payoff functions for option pricing — pure functions, JIT-compatible."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax._util.types import OptionType


# ---------------------------------------------------------------------------
# Vanilla payoffs
# ---------------------------------------------------------------------------

def plain_vanilla_payoff(spot, strike, option_type: int):
    """Standard call/put payoff: max(phi*(S-K), 0)."""
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    return jnp.maximum(phi * (spot - strike), 0.0)


def straddle_payoff(spot, strike):
    """Straddle payoff: |S - K|."""
    return jnp.abs(spot - strike)


# ---------------------------------------------------------------------------
# Binary/digital payoffs
# ---------------------------------------------------------------------------

def cash_or_nothing_payoff(spot, strike, cash_amount, option_type: int):
    """Cash-or-nothing: cash_amount if ITM, else 0."""
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    return jnp.where(phi * (spot - strike) > 0, cash_amount, 0.0)


def asset_or_nothing_payoff(spot, strike, option_type: int):
    """Asset-or-nothing: S if ITM, else 0."""
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    return jnp.where(phi * (spot - strike) > 0, spot, 0.0)


def gap_payoff(spot, strike, second_strike, option_type: int):
    """Gap payoff: max(phi*(S - K2), 0) if phi*(S - K1) > 0."""
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    return jnp.where(phi * (spot - strike) > 0,
                     jnp.maximum(phi * (spot - second_strike), 0.0), 0.0)


def super_share_payoff(spot, strike_low, strike_high):
    """Super-share payoff: S/K_low if K_low < S < K_high, else 0."""
    return jnp.where(
        (spot > strike_low) & (spot < strike_high),
        spot / strike_low, 0.0
    )
