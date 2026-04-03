"""Credit correlation structures."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class BaseCorrelationStructure:
    """Base correlation term structure.

    Maps detachment points to base correlations, optionally across tenors.
    """
    detachment_points: jnp.ndarray  # (n_det,)
    base_correlations: jnp.ndarray  # (n_det,) or (n_tenors, n_det)
    tenors: jnp.ndarray = None  # (n_tenors,) optional

    def correlation(self, detachment, tenor=None):
        """Interpolate base correlation at given detachment point."""
        if self.base_correlations.ndim == 1:
            return float(jnp.interp(detachment, self.detachment_points,
                                     self.base_correlations))
        if tenor is not None and self.tenors is not None:
            # Bilinear interpolation
            corr_at_det = jnp.array([
                float(jnp.interp(detachment, self.detachment_points, row))
                for row in self.base_correlations
            ])
            return float(jnp.interp(tenor, self.tenors, corr_at_det))
        return float(jnp.interp(detachment, self.detachment_points,
                                 self.base_correlations[0]))


@dataclass
class FlatCorrelation:
    """Flat (constant) correlation structure."""
    rho: float

    def correlation(self, *args, **kwargs):
        return self.rho


@dataclass
class PiecewiseCorrelation:
    """Piecewise flat correlation by tenor."""
    tenors: jnp.ndarray  # breakpoints
    values: jnp.ndarray  # correlation values

    def correlation(self, t):
        idx = jnp.searchsorted(self.tenors, t, side='right') - 1
        idx = jnp.clip(idx, 0, len(self.values) - 1)
        return float(self.values[idx])


def factor_spreaded_hazard_rate(base_hazard_rate, spread):
    """Apply a multiplicative factor spread to a hazard rate curve.

    Parameters
    ----------
    base_hazard_rate : base hazard rate(s)
    spread : multiplicative spread factor

    Returns
    -------
    adjusted hazard rate
    """
    return base_hazard_rate * spread
