"""Optimization constraints: bounds, equality, inequality, projections."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import jax.numpy as jnp


@dataclass(frozen=True)
class NoConstraint:
    """Unconstrained (always feasible)."""

    def test(self, x):
        return True

    def upper_bound(self, x):
        return jnp.full_like(x, jnp.inf)

    def lower_bound(self, x):
        return jnp.full_like(x, -jnp.inf)

    def project(self, x):
        return x


@dataclass(frozen=True)
class BoundaryConstraint:
    """Box constraint: lo <= x <= hi."""
    lo: float
    hi: float

    def test(self, x):
        return jnp.all((x >= self.lo) & (x <= self.hi))

    def upper_bound(self, x):
        return jnp.full_like(x, self.hi)

    def lower_bound(self, x):
        return jnp.full_like(x, self.lo)

    def project(self, x):
        return jnp.clip(x, self.lo, self.hi)


@dataclass(frozen=True)
class PositiveConstraint:
    """x > 0 constraint."""

    def test(self, x):
        return jnp.all(x > 0)

    def project(self, x):
        return jnp.maximum(x, 1e-15)


@dataclass(frozen=True)
class NonhomogeneousBoundaryConstraint:
    """Per-component box constraints: lo[i] <= x[i] <= hi[i]."""
    lo: jnp.ndarray
    hi: jnp.ndarray

    def test(self, x):
        return jnp.all((x >= self.lo) & (x <= self.hi))

    def project(self, x):
        return jnp.clip(x, self.lo, self.hi)


@dataclass(frozen=True)
class CompositeConstraint:
    """Intersection of two constraints."""
    c1: object
    c2: object

    def test(self, x):
        return self.c1.test(x) and self.c2.test(x)

    def project(self, x):
        return self.c2.project(self.c1.project(x))


def project_with_constraint(constraint, x):
    """Project x onto the feasible set."""
    return constraint.project(x)
