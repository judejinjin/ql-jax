"""CMS market calibration.

Container for CMS market data (CMS rates, spreads) and calibration
of the CMS coupon model parameters to market quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class CmsMarket:
    """Container for CMS market data.

    Parameters
    ----------
    tenors : array of swap tenors (years)
    forward_rates : CMS forward rates for each tenor
    vol_atm : ATM swaption vols for each tenor
    mean_reversions : CMS convexity adjustment parameters
    """
    tenors: jnp.ndarray
    forward_rates: jnp.ndarray
    vol_atm: jnp.ndarray
    mean_reversions: jnp.ndarray = None

    def cms_rate(self, tenor_idx, adjustment=True):
        """CMS rate with optional convexity adjustment.

        Parameters
        ----------
        tenor_idx : index into tenors array
        adjustment : include convexity adjustment

        Returns
        -------
        rate : CMS rate
        """
        fwd = float(self.forward_rates[tenor_idx])
        if not adjustment:
            return fwd

        vol = float(self.vol_atm[tenor_idx])
        T = float(self.tenors[tenor_idx])
        # First-order convexity adjustment: CMS ≈ swap + vol²*T*swap²/annuity_duration
        # Simplified: CMS ≈ swap + vol² * T * swap (proportional)
        return fwd * (1.0 + vol**2 * T)


def calibrate_cms_model(market, target_cms_rates, mean_reversion_guess=0.01):
    """Calibrate mean reversion parameters to match CMS rates.

    Parameters
    ----------
    market : CmsMarket
    target_cms_rates : observed CMS rates
    mean_reversion_guess : initial guess

    Returns
    -------
    mean_reversions : calibrated mean reversion for each tenor
    """
    target = jnp.asarray(target_cms_rates, dtype=jnp.float64)
    n = len(market.tenors)

    # Simple calibration: adjust mean reversion to match CMS rate
    mrs = []
    for i in range(n):
        fwd = float(market.forward_rates[i])
        vol = float(market.vol_atm[i])
        T = float(market.tenors[i])
        tgt = float(target[i])

        # CMS(a) ≈ fwd * (1 + vol² * T / (1 + a*T))
        # Solve: tgt = fwd * (1 + vol²*T/(1+a*T))
        # => a = (fwd*vol²*T/(tgt - fwd) - 1) / T
        if abs(tgt - fwd) > 1e-12:
            a = (fwd * vol**2 * T / (tgt - fwd) - 1.0) / T
            a = max(a, 0.0)
        else:
            a = mean_reversion_guess
        mrs.append(a)

    return jnp.array(mrs)
