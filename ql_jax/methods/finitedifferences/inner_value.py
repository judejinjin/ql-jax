"""Inner value calculators – evaluating payoffs on FD grids."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class VanillaInnerValue:
    """Vanilla option payoff on a grid.

    Parameters
    ----------
    strike : float
    is_call : bool
    grid : array – spatial grid (spot or log-spot)
    is_log : bool – if True, grid is in log-spot space
    """
    strike: float
    is_call: bool = True
    grid: jnp.ndarray = None
    is_log: bool = False

    def values(self, grid=None):
        """Compute intrinsic values on the grid."""
        g = grid if grid is not None else self.grid
        s = jnp.exp(g) if self.is_log else g
        phi = 1.0 if self.is_call else -1.0
        return jnp.maximum(phi * (s - self.strike), 0.0)


@dataclass(frozen=True)
class BarrierInnerValue:
    """Barrier option payoff – zero beyond barrier.

    Parameters
    ----------
    strike : float
    barrier : float
    is_call : bool
    is_up_and_out : bool
    rebate : float
    grid : array
    is_log : bool
    """
    strike: float
    barrier: float
    is_call: bool = True
    is_up_and_out: bool = True
    rebate: float = 0.0
    grid: jnp.ndarray = None
    is_log: bool = False

    def values(self, grid=None):
        g = grid if grid is not None else self.grid
        s = jnp.exp(g) if self.is_log else g
        phi = 1.0 if self.is_call else -1.0
        vanilla = jnp.maximum(phi * (s - self.strike), 0.0)

        if self.is_up_and_out:
            knocked = s >= self.barrier
        else:
            knocked = s <= self.barrier

        return jnp.where(knocked, self.rebate, vanilla)


@dataclass(frozen=True)
class DigitalInnerValue:
    """Digital option payoff: fixed payout if in-the-money.

    Parameters
    ----------
    strike : float
    payout : float
    is_call : bool
    grid : array
    is_log : bool
    """
    strike: float
    payout: float = 1.0
    is_call: bool = True
    grid: jnp.ndarray = None
    is_log: bool = False

    def values(self, grid=None):
        g = grid if grid is not None else self.grid
        s = jnp.exp(g) if self.is_log else g
        if self.is_call:
            return jnp.where(s >= self.strike, self.payout, 0.0)
        else:
            return jnp.where(s <= self.strike, self.payout, 0.0)


@dataclass(frozen=True)
class StraddleInnerValue:
    """Straddle payoff: |S - K|."""
    strike: float
    grid: jnp.ndarray = None
    is_log: bool = False

    def values(self, grid=None):
        g = grid if grid is not None else self.grid
        s = jnp.exp(g) if self.is_log else g
        return jnp.abs(s - self.strike)
