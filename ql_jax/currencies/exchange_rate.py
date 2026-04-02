"""Exchange rate manager and FX conversion."""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import jax.numpy as jnp

from ql_jax.currencies.america import Currency


class ExchangeRate:
    """A single exchange rate between two currencies.

    Parameters
    ----------
    source : Currency
    target : Currency
    rate : float (1 unit of source = rate units of target)
    """

    def __init__(self, source: Currency, target: Currency, rate: float):
        self.source = source
        self.target = target
        self.rate = jnp.float64(rate)

    def exchange(self, amount: float) -> float:
        """Convert amount from source to target currency."""
        return amount * self.rate

    def inverse(self) -> 'ExchangeRate':
        """Return the inverse exchange rate."""
        return ExchangeRate(self.target, self.source, 1.0 / self.rate)

    def __repr__(self):
        return f"{self.source.code}/{self.target.code} = {self.rate:.6f}"


class ExchangeRateManager:
    """Global store for exchange rates with triangulation support."""

    _instance = None
    _rates: Dict[Tuple[str, str], ExchangeRate] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._rates = {}
        return cls._instance

    def add(self, rate: ExchangeRate):
        """Add an exchange rate (and its inverse)."""
        self._rates[(rate.source.code, rate.target.code)] = rate
        self._rates[(rate.target.code, rate.source.code)] = rate.inverse()

    def lookup(self, source_code: str, target_code: str) -> Optional[ExchangeRate]:
        """Look up a direct or triangulated exchange rate.

        Parameters
        ----------
        source_code, target_code : str (ISO currency codes)

        Returns
        -------
        ExchangeRate or None
        """
        if source_code == target_code:
            from ql_jax.currencies.america import USD
            return ExchangeRate(USD, USD, 1.0)

        # Direct
        key = (source_code, target_code)
        if key in self._rates:
            return self._rates[key]

        # Triangulate via USD
        k1 = (source_code, "USD")
        k2 = ("USD", target_code)
        if k1 in self._rates and k2 in self._rates:
            r1 = self._rates[k1]
            r2 = self._rates[k2]
            return ExchangeRate(r1.source, r2.target, r1.rate * r2.rate)

        # Triangulate via EUR
        k1 = (source_code, "EUR")
        k2 = ("EUR", target_code)
        if k1 in self._rates and k2 in self._rates:
            r1 = self._rates[k1]
            r2 = self._rates[k2]
            return ExchangeRate(r1.source, r2.target, r1.rate * r2.rate)

        return None

    def clear(self):
        """Clear all rates."""
        self._rates.clear()
