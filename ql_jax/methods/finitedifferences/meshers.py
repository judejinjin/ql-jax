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


# ---------------------------------------------------------------------------
# Black-Scholes mesher  (log-spot concentrating near strike)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FdmBlackScholesMesher:
    """Black-Scholes mesher concentrating around strike.

    Parameters
    ----------
    size : number of grid points
    strike : concentration point
    spot : current spot price
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    maturity : time to maturity
    stdev_range : range in standard deviations
    """
    size: int
    strike: float
    spot: float
    r: float = 0.05
    q: float = 0.0
    sigma: float = 0.2
    maturity: float = 1.0
    stdev_range: float = 4.0

    def locations(self):
        log_fwd = jnp.log(self.spot) + (self.r - self.q - 0.5 * self.sigma ** 2) * self.maturity
        stdev = self.sigma * jnp.sqrt(self.maturity) * self.stdev_range
        low = log_fwd - stdev
        high = log_fwd + stdev
        center = jnp.log(self.strike)
        mesher = Concentrating1dMesher(low, high, self.size, center, density=5.0)
        return mesher.locations()

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class FdmHestonVarianceMesher:
    """Variance mesher for Heston model.

    Parameters
    ----------
    size : grid points
    v0 : initial variance
    kappa : mean reversion
    theta : long-run variance
    sigma_v : vol of vol
    maturity : time in years
    """
    size: int
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    maturity: float

    def locations(self):
        v_max = self.theta + 4.0 * self.sigma_v * jnp.sqrt(self.theta / (2.0 * self.kappa))
        v_max = jnp.maximum(v_max, 2.0 * self.v0)
        return Concentrating1dMesher(
            1e-6, v_max, self.size, self.v0, density=3.0,
        ).locations()

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


# ---------------------------------------------------------------------------
# Additional meshers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predefined1dMesher:
    """User-supplied grid points.

    Parameters
    ----------
    grid : array of sorted grid locations
    """
    grid: jnp.ndarray

    @property
    def size(self):
        return self.grid.shape[0]

    @property
    def low(self):
        return float(self.grid[0])

    @property
    def high(self):
        return float(self.grid[-1])

    def locations(self):
        return self.grid

    def dminus(self):
        return jnp.diff(self.grid)

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class ExponentialJump1dMesher:
    """Mesher for jump-diffusion: concentrates near log-spot with exponential tails.

    Parameters
    ----------
    low, high : bounds
    size : grid points
    center : concentration center
    beta : exponential decay parameter
    """
    low: float
    high: float
    size: int
    center: float
    beta: float = 1.0

    def locations(self):
        xi = jnp.linspace(0.0, 1.0, self.size)
        # Map through double-exponential
        c = (self.high - self.low) / 2.0
        x = self.center + c * jnp.tanh(self.beta * (2.0 * xi - 1.0))
        return x

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class FdmCEV1dMesher:
    """1D mesher for the CEV process dS = (r-q)*S*dt + sigma*S^beta*dW.

    Concentrates grid points near the origin (where S^beta behaviour
    changes) and near the spot.
    """
    low: float
    high: float
    size: int
    spot: float = 100.0
    sigma: float = 0.2
    beta: float = 0.5

    def locations(self):
        u = jnp.linspace(0.0, 1.0, self.size)
        exponent = max(1.0 / (2.0 - 2.0 * self.beta + 0.01), 1.0)
        s = u ** exponent
        return self.low + (self.high - self.low) * s

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()


@dataclass(frozen=True)
class FdmBlackScholesMultiStrikeMesher:
    """1D mesher for BS that concentrates around multiple strike levels.

    Useful for basket / rainbow options with multiple payoff boundaries.
    """
    low: float
    high: float
    size: int
    strikes: tuple = ()
    density: float = 5.0

    def locations(self):
        if not self.strikes:
            return jnp.linspace(self.low, self.high, self.size)

        u = jnp.linspace(0.0, 1.0, self.size)
        rng = self.high - self.low
        x = self.low + rng * u

        for K in self.strikes:
            c = (K - self.low) / rng
            shift = self.density * (u - c)
            mapped = c + jnp.arcsinh(shift) / (2.0 * self.density)
            x = 0.5 * x + 0.5 * (self.low + rng * jnp.clip(mapped, 0.0, 1.0))

        return jnp.sort(x)

    def dminus(self):
        return jnp.diff(self.locations())

    def dplus(self):
        return self.dminus()
