"""Advanced statistics: incremental, convergence, discrepancy, sequence, histogram."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class IncrementalStatistics:
    """Online (streaming) statistics using Welford's algorithm.

    Maintains running mean, variance, min, max without storing all data.
    """
    _n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0
    _min: float = float('inf')
    _max: float = float('-inf')

    def add(self, x):
        """Add a single observation."""
        x = float(x)
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        self._min = min(self._min, x)
        self._max = max(self._max, x)

    def add_batch(self, xs):
        """Add multiple observations."""
        for x in xs:
            self.add(x)

    @property
    def count(self):
        return self._n

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._m2 / max(self._n - 1, 1)

    @property
    def std(self):
        return self.variance ** 0.5

    @property
    def min_value(self):
        return self._min

    @property
    def max_value(self):
        return self._max

    @property
    def error_estimate(self):
        """Standard error of the mean."""
        return self.std / max(self._n, 1) ** 0.5


@dataclass
class ConvergenceStatistics:
    """Track convergence of an iterative computation.

    Stores the running mean at specified sample sizes.
    """
    _stats: IncrementalStatistics = field(default_factory=IncrementalStatistics)
    _checkpoints: list = field(default_factory=list)
    _check_interval: int = 100

    def __init__(self, check_interval=100):
        self._stats = IncrementalStatistics()
        self._checkpoints = []
        self._check_interval = check_interval

    def add(self, x):
        self._stats.add(x)
        if self._stats.count % self._check_interval == 0:
            self._checkpoints.append({
                'n': self._stats.count,
                'mean': self._stats.mean,
                'std_error': self._stats.error_estimate,
            })

    @property
    def convergence_table(self):
        return self._checkpoints

    @property
    def stats(self):
        return self._stats


def discrepancy(points):
    """Star discrepancy estimate for a point set in [0,1]^d.

    Uses the L2-star discrepancy formula.

    Parameters
    ----------
    points : array [n_points, n_dims] in [0, 1]^d

    Returns
    -------
    float : L2-star discrepancy
    """
    points = jnp.asarray(points, dtype=jnp.float64)
    n, d = points.shape

    # L2-star discrepancy
    # D^2 = (1/3)^d - (2/n) * sum_i prod_k (1 - x_{ik}^2)/2
    #      + (1/n^2) * sum_{i,j} prod_k (1 - max(x_{ik}, x_{jk}))

    term1 = (1.0 / 3.0) ** d

    prod_term = jnp.prod((1.0 - points ** 2) / 2.0, axis=1)
    term2 = -2.0 / n * jnp.sum(prod_term)

    # Pairwise term (expensive for large n)
    term3 = 0.0
    for i in range(n):
        maxes = jnp.maximum(points[i:i + 1, :], points)
        prod_min = jnp.prod(1.0 - maxes, axis=1)
        term3 = term3 + jnp.sum(prod_min)
    term3 = term3 / n ** 2

    return jnp.sqrt(jnp.maximum(term1 + term2 + term3, 0.0))


@dataclass
class SequenceStatistics:
    """Multi-dimensional statistics for sequence data.

    Parameters
    ----------
    dimension : int
    """
    dimension: int
    _n: int = 0
    _sum: jnp.ndarray = None
    _sum_sq: jnp.ndarray = None
    _correlation_sum: jnp.ndarray = None

    def __post_init__(self):
        self._sum = jnp.zeros(self.dimension, dtype=jnp.float64)
        self._sum_sq = jnp.zeros(self.dimension, dtype=jnp.float64)
        self._correlation_sum = jnp.zeros((self.dimension, self.dimension), dtype=jnp.float64)

    def add(self, x):
        """Add a d-dimensional observation."""
        x = jnp.asarray(x, dtype=jnp.float64)
        self._n += 1
        self._sum = self._sum + x
        self._sum_sq = self._sum_sq + x ** 2
        self._correlation_sum = self._correlation_sum + jnp.outer(x, x)

    @property
    def count(self):
        return self._n

    @property
    def mean(self):
        return self._sum / max(self._n, 1)

    @property
    def variance(self):
        n = max(self._n, 2)
        return (self._sum_sq / n - self.mean ** 2) * n / (n - 1)

    @property
    def correlation(self):
        """Correlation matrix."""
        n = max(self._n, 2)
        cov = self._correlation_sum / n - jnp.outer(self.mean, self.mean)
        std = jnp.sqrt(jnp.maximum(jnp.diag(cov), 1e-15))
        return cov / jnp.outer(std, std)


class Histogram:
    """Simple histogram class.

    Parameters
    ----------
    bins : int or array of bin edges
    range_ : (min, max) tuple (optional if bins is array)
    """

    def __init__(self, bins=50, range_=None):
        self._bin_spec = bins
        self._range = range_
        self._counts = None
        self._edges = None

    def add(self, data):
        """Add data to histogram."""
        data = jnp.asarray(data, dtype=jnp.float64)
        if self._range is not None:
            counts, edges = jnp.histogram(data, bins=self._bin_spec, range=self._range)
        else:
            counts, edges = jnp.histogram(data, bins=self._bin_spec)

        if self._counts is None:
            self._counts = counts
            self._edges = edges
        else:
            self._counts = self._counts + counts

    @property
    def counts(self):
        return self._counts

    @property
    def edges(self):
        return self._edges

    @property
    def bin_centers(self):
        return 0.5 * (self._edges[:-1] + self._edges[1:])

    @property
    def density(self):
        """Normalized density."""
        dx = jnp.diff(self._edges)
        total = jnp.sum(self._counts * dx)
        return self._counts / jnp.maximum(total, 1e-15)
