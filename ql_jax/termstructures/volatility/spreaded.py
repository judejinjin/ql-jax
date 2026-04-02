"""Spreaded volatility structures: smile sections, optionlet, and swaption vols."""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SpreadedSmileSection:
    """Smile section with a parallel vol spread.

    Parameters
    ----------
    base_smile_fn : callable(K) -> vol
    spread : float – additive vol spread
    """
    base_smile_fn: Callable
    spread: float

    def volatility(self, strike):
        return self.base_smile_fn(strike) + self.spread

    def __call__(self, strike):
        return self.volatility(strike)


@dataclass(frozen=True)
class SpreadedOptionletVolatility:
    """Optionlet vol surface with a term structure of spreads.

    Parameters
    ----------
    base_vol_fn : callable(T, K) -> vol
    spread_fn : callable(T) -> spread
    """
    base_vol_fn: Callable
    spread_fn: Callable

    def volatility(self, T, K):
        return self.base_vol_fn(T, K) + self.spread_fn(T)


@dataclass(frozen=True)
class SpreadedSwaptionVolatility:
    """Swaption vol cube with a spread.

    Parameters
    ----------
    base_vol_fn : callable(expiry, tenor, strike) -> vol
    spread : float or callable(expiry, tenor) -> spread
    """
    base_vol_fn: Callable
    spread: object  # float or callable

    def volatility(self, expiry, tenor, strike):
        base = self.base_vol_fn(expiry, tenor, strike)
        if callable(self.spread):
            return base + self.spread(expiry, tenor)
        return base + self.spread
