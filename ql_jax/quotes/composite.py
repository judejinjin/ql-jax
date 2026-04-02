"""Composite and derived quote types."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


class DerivedQuote:
    """A quote whose value is f(base_quote).

    Parameters
    ----------
    base_quote : object with .value attribute (e.g., SimpleQuote)
    transform : callable(float) -> float
    """

    def __init__(self, base_quote, transform: Callable):
        self._base = base_quote
        self._fn = transform

    @property
    def value(self):
        return self._fn(self._base.value)


class CompositeQuote:
    """A quote whose value is f(quote1, quote2).

    Parameters
    ----------
    quote1, quote2 : objects with .value attribute
    compose : callable(float, float) -> float
    """

    def __init__(self, quote1, quote2, compose: Callable):
        self._q1 = quote1
        self._q2 = quote2
        self._fn = compose

    @property
    def value(self):
        return self._fn(self._q1.value, self._q2.value)


class DeltaVolQuote:
    """Quote representing a volatility at a given delta.

    Parameters
    ----------
    delta : float (e.g., 0.25 for 25-delta)
    vol : float
    maturity : float (year fraction)
    at_the_money_type : str ('forward', 'spot')
    """

    def __init__(self, delta, vol, maturity, at_the_money_type='forward'):
        self._delta = delta
        self._vol = vol
        self._maturity = maturity
        self._atm_type = at_the_money_type

    @property
    def value(self):
        return self._vol

    @property
    def delta(self):
        return self._delta

    @property
    def maturity(self):
        return self._maturity


class ForwardValueQuote:
    """Quote for a forward value derived from a term structure.

    Parameters
    ----------
    index : object with forward_rate(t) or fixing(t)
    fixing_date_yearfrac : float
    """

    def __init__(self, index, fixing_date_yearfrac):
        self._index = index
        self._t = fixing_date_yearfrac

    @property
    def value(self):
        if hasattr(self._index, 'forward_rate'):
            return self._index.forward_rate(self._t)
        return self._index.fixing(self._t)


class ForwardSwapQuote:
    """Quote for a forward swap rate.

    Parameters
    ----------
    swap_index : SwapIndex
    fixing_date_yearfrac : float
    spread : float
    """

    def __init__(self, swap_index, fixing_date_yearfrac, spread=0.0):
        self._index = swap_index
        self._t = fixing_date_yearfrac
        self._spread = spread

    @property
    def value(self):
        return self._index.fixing(self._t) + self._spread


class ImpliedStdDevQuote:
    """Quote for an implied standard deviation.

    Parameters
    ----------
    option_type : str ('call' or 'put')
    forward : float
    strike : float
    price_quote : object with .value
    discount : float
    """

    def __init__(self, option_type, forward, strike, price_quote, discount=1.0):
        self._type = option_type
        self._fwd = forward
        self._K = strike
        self._price = price_quote
        self._df = discount

    @property
    def value(self):
        """Implied vol * sqrt(T), i.e., implied standard deviation."""
        from ql_jax.engines.analytic.black_formula import implied_volatility_black_scholes
        price = self._price.value
        # Use zero expiry as placeholder — caller supplies meaningful T
        try:
            return implied_volatility_black_scholes(
                price, self._fwd, self._K, 1.0, 0.0, 0.0,
                1 if self._type == 'call' else -1,
            )
        except Exception:
            return 0.0


class EurodollarFuturesQuote:
    """Quote for a Eurodollar futures contract.

    Value = 100 - implied_rate.

    Parameters
    ----------
    rate_quote : object with .value (forward rate)
    """

    def __init__(self, rate_quote):
        self._rate = rate_quote

    @property
    def value(self):
        return 100.0 - self._rate.value * 100.0


class FuturesConvAdjustmentQuote:
    """Quote applying a convexity adjustment to a futures rate.

    Parameters
    ----------
    futures_quote : object with .value (futures rate)
    adjustment : float (convexity adjustment to subtract)
    """

    def __init__(self, futures_quote, adjustment=0.0):
        self._futures = futures_quote
        self._adj = adjustment

    @property
    def value(self):
        return self._futures.value - self._adj
