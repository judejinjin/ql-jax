"""FX indexes and equity indexes."""

from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass
class FXIndex:
    """Foreign exchange rate index.

    Parameters
    ----------
    source_currency : str
    target_currency : str
    fixing_days : int
    source_curve : callable or None – yield curve for source currency
    target_curve : callable or None – yield curve for target currency
    """
    source_currency: str
    target_currency: str
    fixing_days: int = 2
    source_curve: object = None
    target_curve: object = None
    _fixings: dict = field(default_factory=dict, repr=False)

    def name(self):
        return f"{self.source_currency}{self.target_currency}"

    def add_fixing(self, date_serial, value):
        self._fixings[date_serial] = value

    def fixing(self, date_serial):
        return self._fixings.get(date_serial, None)

    def forward_rate(self, T):
        """Forward FX rate via covered interest parity.

        F = S * P_target(T) / P_source(T)
        """
        if self.source_curve is None or self.target_curve is None:
            return None
        spot = list(self._fixings.values())[-1] if self._fixings else 1.0
        return spot * self.target_curve(T) / self.source_curve(T)


@dataclass
class EquityIndex:
    """Equity index for equity-linked products.

    Parameters
    ----------
    name_ : str
    currency : str
    dividend_yield : float
    """
    name_: str
    currency: str
    dividend_yield: float = 0.0
    _fixings: dict = field(default_factory=dict, repr=False)

    def name(self):
        return self.name_

    def add_fixing(self, date_serial, value):
        self._fixings[date_serial] = value

    def fixing(self, date_serial):
        return self._fixings.get(date_serial, None)

    def forward_price(self, spot, r, T):
        """Forward equity price."""
        return spot * jnp.exp((r - self.dividend_yield) * T)


# --- Standard FX pairs ---

def EURUSD(fixing_days=2):
    return FXIndex("EUR", "USD", fixing_days)

def USDJPY(fixing_days=2):
    return FXIndex("USD", "JPY", fixing_days)

def GBPUSD(fixing_days=2):
    return FXIndex("GBP", "USD", fixing_days)

def USDCHF(fixing_days=2):
    return FXIndex("USD", "CHF", fixing_days)

def AUDUSD(fixing_days=2):
    return FXIndex("AUD", "USD", fixing_days)

def USDCAD(fixing_days=2):
    return FXIndex("USD", "CAD", fixing_days)

def EURGBP(fixing_days=2):
    return FXIndex("EUR", "GBP", fixing_days)


# --- Standard equity indexes ---

def SPX():
    return EquityIndex("S&P 500", "USD", 0.015)

def EUROSTOXX50():
    return EquityIndex("Euro Stoxx 50", "EUR", 0.03)

def FTSE100():
    return EquityIndex("FTSE 100", "GBP", 0.035)

def NIKKEI225():
    return EquityIndex("Nikkei 225", "JPY", 0.02)

def DAX():
    return EquityIndex("DAX", "EUR", 0.0)  # Total return
