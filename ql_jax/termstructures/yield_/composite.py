"""Composite zero-yield term structure.

Combines two yield curves via an arithmetic or geometric operation:
  y_composite(t) = op(y1(t), y2(t))
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CompositeZeroYieldStructure:
    """Composite of two zero-yield curves.

    Parameters
    ----------
    curve1_fn : callable(t) -> zero rate
    curve2_fn : callable(t) -> zero rate
    compose_fn : callable(r1, r2) -> r, default addition
    """
    curve1_fn: Callable
    curve2_fn: Callable
    compose_fn: Callable = None

    def zero_rate(self, t):
        r1 = self.curve1_fn(t)
        r2 = self.curve2_fn(t)
        if self.compose_fn is not None:
            return self.compose_fn(r1, r2)
        return r1 + r2

    def discount(self, t):
        return jnp.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1, t2):
        d1 = self.discount(t1)
        d2 = self.discount(t2)
        return -jnp.log(d2 / d1) / (t2 - t1)
