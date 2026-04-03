"""Energy/commodities framework — commodity types, instruments, processes."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional


# ── Commodity types and settings ──

@dataclass
class CommodityType:
    """Commodity type definition."""
    name: str  # e.g. "NaturalGas", "Crude", "Power"
    unit: str = "MWh"  # unit of measure

    def __hash__(self):
        return hash(self.name)


@dataclass
class CommodityIndex:
    """Commodity price index — reference for pricing."""
    name: str
    commodity_type: CommodityType
    currency: str = "USD"

    def __hash__(self):
        return hash(self.name)


@dataclass
class CommodityCurve:
    """Commodity forward curve.

    Parameters
    ----------
    times : (n,) times to delivery (years)
    prices : (n,) forward prices
    """
    times: jnp.ndarray
    prices: jnp.ndarray

    def forward_price(self, t):
        """Interpolate forward price at time t."""
        return float(jnp.interp(t, self.times, self.prices))


# ── Instruments ──

@dataclass
class EnergyFuture:
    """Energy futures contract.

    Parameters
    ----------
    commodity_index : CommodityIndex
    delivery_start : start of delivery period (years)
    delivery_end : end of delivery period (years)
    quantity : contract quantity
    """
    commodity_index: CommodityIndex
    delivery_start: float
    delivery_end: float
    quantity: float = 1.0

    def fair_value(self, curve):
        """Fair value using forward curve."""
        t_mid = (self.delivery_start + self.delivery_end) / 2
        return curve.forward_price(t_mid) * self.quantity


@dataclass
class EnergySwap:
    """Energy swap — exchange of floating commodity price for fixed price.

    Parameters
    ----------
    commodity_index : CommodityIndex
    fixed_price : fixed leg price
    observation_times : (n_obs,) times of floating price observations
    quantity : per-period quantity
    """
    commodity_index: CommodityIndex
    fixed_price: float
    observation_times: jnp.ndarray
    quantity: float = 1.0

    def fair_value(self, curve, discount_factors):
        """Fair value of the swap (from receiver of floating perspective)."""
        obs_t = jnp.asarray(self.observation_times)
        df = jnp.asarray(discount_factors)
        floating_leg = 0.0
        fixed_leg = 0.0
        for i in range(len(obs_t)):
            fwd = curve.forward_price(float(obs_t[i]))
            d = float(df[i])
            floating_leg += fwd * self.quantity * d
            fixed_leg += self.fixed_price * self.quantity * d
        return floating_leg - fixed_leg


@dataclass
class EnergyBasisSwap:
    """Basis swap between two commodity indexes.

    Parameters
    ----------
    index1, index2 : CommodityIndex objects
    spread : fixed spread (index1 - index2 - spread = 0 at fair value)
    observation_times : observation dates
    """
    index1: CommodityIndex
    index2: CommodityIndex
    spread: float
    observation_times: jnp.ndarray
    quantity: float = 1.0

    def fair_value(self, curve1, curve2, discount_factors):
        """Fair value of basis swap."""
        obs_t = self.observation_times
        df = jnp.asarray(discount_factors)
        val = 0.0
        for i in range(len(obs_t)):
            fwd1 = curve1.forward_price(float(obs_t[i]))
            fwd2 = curve2.forward_price(float(obs_t[i]))
            d = float(df[i])
            val += (fwd1 - fwd2 - self.spread) * self.quantity * d
        return val


# ── Commodity cash flows ──

@dataclass
class CommodityCashFlow:
    """A single commodity-linked cash flow."""
    payment_time: float
    quantity: float
    commodity_index: CommodityIndex

    def amount(self, curve):
        return curve.forward_price(self.payment_time) * self.quantity


# ── Processes ──

def geman_roncoroni_process(S0, mu, sigma, alpha, lam_up, lam_down,
                             eta_up, eta_down, T, n_steps, n_paths, seed=42):
    """Geman-Roncoroni mean-reverting jump-diffusion for energy prices.

    dS = alpha*(mu - S) dt + sigma dW + J_up dN_up - J_down dN_down

    Parameters
    ----------
    S0 : initial price
    mu : long-term mean level
    sigma : diffusion volatility
    alpha : mean-reversion speed
    lam_up, lam_down : Poisson intensities for up/down jumps
    eta_up, eta_down : exponential jump size parameters
    T : time horizon
    n_steps : number of time steps
    n_paths : number of simulated paths
    seed : RNG seed

    Returns
    -------
    t_grid : (n_steps+1,) time grid
    paths : (n_paths, n_steps+1) simulated paths
    """
    dt = T / n_steps
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    dW = jax.random.normal(k1, (n_paths, n_steps)) * jnp.sqrt(dt)
    N_up = jax.random.poisson(k2, lam_up * dt, (n_paths, n_steps))
    N_down = jax.random.poisson(k3, lam_down * dt, (n_paths, n_steps))
    J_up = jax.random.exponential(k4, (n_paths, n_steps)) / eta_up
    J_down = jax.random.exponential(k5, (n_paths, n_steps)) / eta_down

    paths = jnp.zeros((n_paths, n_steps + 1))
    paths = paths.at[:, 0].set(S0)
    for i in range(n_steps):
        S = paths[:, i]
        dS = (alpha * (mu - S) * dt + sigma * dW[:, i] +
              J_up[:, i] * N_up[:, i] - J_down[:, i] * N_down[:, i])
        paths = paths.at[:, i + 1].set(jnp.maximum(S + dS, 0.0))

    t_grid = jnp.linspace(0, T, n_steps + 1)
    return t_grid, paths
