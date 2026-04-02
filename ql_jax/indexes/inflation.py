"""Inflation indexes – CPI, HICP, RPI, etc."""

from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass
class InflationIndex:
    """Base class for inflation indexes.

    Parameters
    ----------
    family_name : str
    region : str
    revised : bool – whether fixings can be revised
    frequency : str – 'Monthly' or 'Daily'
    observation_lag : int – months of lag
    """
    family_name: str
    region: str
    revised: bool = False
    frequency: str = 'Monthly'
    observation_lag: int = 3
    _fixings: dict = field(default_factory=dict, repr=False)

    def name(self):
        return f"{self.family_name} {self.region}"

    def add_fixing(self, date_serial, value):
        """Record a historical fixing."""
        self._fixings[date_serial] = value

    def fixing(self, date_serial):
        """Return historical or interpolated fixing."""
        return self._fixings.get(date_serial, None)


@dataclass
class ZeroInflationIndex(InflationIndex):
    """Zero-coupon inflation index (e.g., CPI, HICP).

    Used for zero-coupon inflation swaps and bonds.
    """
    pass


@dataclass
class YoYInflationIndex(InflationIndex):
    """Year-on-year inflation index.

    Returns the year-on-year ratio of the underlying ZC index.
    """
    underlying: InflationIndex = None

    def yoy_rate(self, date_serial, lag_months=12):
        """Compute YoY rate: I(t) / I(t-12m) - 1."""
        if self.underlying is None:
            return None
        i_t = self.underlying.fixing(date_serial)
        i_prev = self.underlying.fixing(date_serial - lag_months * 30)  # approx
        if i_t is not None and i_prev is not None:
            return i_t / i_prev - 1.0
        return None


# --- Standard inflation indexes ---

def USCPI():
    """US CPI-U (Urban consumers)."""
    return ZeroInflationIndex(
        family_name="CPI", region="US", revised=False,
        frequency='Monthly', observation_lag=2,
    )


def UKRPI():
    """UK Retail Price Index."""
    return ZeroInflationIndex(
        family_name="RPI", region="UK", revised=True,
        frequency='Monthly', observation_lag=2,
    )


def EUHICP():
    """Eurozone HICP (Harmonised Index of Consumer Prices)."""
    return ZeroInflationIndex(
        family_name="HICP", region="EU", revised=True,
        frequency='Monthly', observation_lag=3,
    )


def FRHICP():
    """French HICP."""
    return ZeroInflationIndex(
        family_name="HICP", region="FR", revised=True,
        frequency='Monthly', observation_lag=3,
    )


def JPCPI():
    """Japan CPI."""
    return ZeroInflationIndex(
        family_name="CPI", region="JP", revised=False,
        frequency='Monthly', observation_lag=3,
    )


def AUCPI():
    """Australian CPI."""
    return ZeroInflationIndex(
        family_name="CPI", region="AU", revised=False,
        frequency='Monthly', observation_lag=2,
    )
