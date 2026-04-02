"""Finite-difference meshers – 1D grids for PDE spatial discretization."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class Uniform1dMesher:
    """Uniform 1D mesh.

    Parameters
    ----------
    low : float – lower bound
    high : float – upper bound
    size : int – number of grid points
    """
    low: float
    high: float
    size: int

    def locations(self):
        return jnp.linspace(self.low, self.high, self.size)

    def dx(self):
        return (self.high - self.low) / (self.size - 1)

    def dminus(self):
        """Backward spacing: x[i] - x[i-1]."""
        x = self.locations()
        return jnp.diff(x)

    def dplus(self):
        """Forward spacing: x[i+1] - x[i]."""
        return self.dminus()


@dataclass(frozen=True)
class Concentrating1dMesher:
    """Concentrating 1D mesh – denser near a focal point.

    Uses sinh transformation for concentration.

    Parameters
    ----------
    low : float
    high : float
    size : int
    center : float – focal point for concentration
    density : float – concentration factor (higher = more points near center)
    """
    low: float
    high: float
    size: int
    center: float
    density: float = 10.0

    def locations(self):
        n = self.size
        xi = jnp.linspace(0.0, 1.0, n)

        # Sinh transformation for concentration
        c1 = jnp.arcsinh((self.low - self.center) * self.density)
        c2 = jnp.arcsinh((self.high - self.center) * self.density)

        x = self.center + jnp.sinh(c1 + (c2 - c1) * xi) / self.density
        return x

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class LogMesher:
    """Logarithmic mesher – uniform in log-space.

    Parameters
    ----------
    low : float – lower bound (must be > 0)
    high : float – upper bound
    size : int
    """
    low: float
    high: float
    size: int

    def locations(self):
        return jnp.exp(jnp.linspace(jnp.log(self.low), jnp.log(self.high), self.size))

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class FdmMesherComposite:
    """Multi-dimensional composite mesher.

    Combines 1D meshers for each dimension.

    Parameters
    ----------
    meshers : tuple of 1D meshers
    """
    meshers: tuple

    @property
    def ndim(self):
        return len(self.meshers)

    def locations(self, dim):
        """Grid points for dimension dim."""
        return self.meshers[dim].locations()

    def sizes(self):
        """Number of points per dimension."""
        return tuple(m.size for m in self.meshers)

    def total_size(self):
        """Total number of grid points (product of sizes)."""
        s = 1
        for m in self.meshers:
            s *= m.size
        return s

    def grid_arrays(self):
        """Return meshgrid arrays for all dimensions."""
        locs = [m.locations() for m in self.meshers]
        return jnp.meshgrid(*locs, indexing='ij')
