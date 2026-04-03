"""Catastrophe bond framework — risk models and MC pricing."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CatRisk:
    """Base catastrophe risk specification."""
    pass


@dataclass
class BetaRisk(CatRisk):
    """Beta-distributed catastrophe loss.

    Loss severity is Beta(alpha, beta) distributed.
    Events arrive with Poisson intensity.

    Parameters
    ----------
    alpha, beta : Beta distribution parameters
    intensity : Poisson event intensity (per year)
    """
    alpha: float
    beta: float
    intensity: float = 1.0


@dataclass
class EventSetRisk(CatRisk):
    """Event-set based catastrophe risk.

    Each event has a probability and loss severity.

    Parameters
    ----------
    event_probs : (n_events,) annual probability of each event
    event_losses : (n_events,) loss given event (as fraction of notional)
    """
    event_probs: jnp.ndarray
    event_losses: jnp.ndarray


@dataclass
class CatBond:
    """Catastrophe bond instrument.

    Parameters
    ----------
    notional : bond notional
    coupon_rate : coupon rate (spread above risk-free)
    maturity : years to maturity
    attachment : loss threshold to trigger principal erosion
    exhaustion : loss threshold for full principal loss
    risk : CatRisk specification
    """
    notional: float
    coupon_rate: float
    maturity: float
    attachment: float
    exhaustion: float
    risk: CatRisk

    @property
    def width(self):
        return self.exhaustion - self.attachment


def mc_cat_bond_price(bond, r, n_paths=50000, n_steps_per_year=12, seed=42):
    """Monte Carlo pricing for catastrophe bonds.

    Parameters
    ----------
    bond : CatBond
    r : risk-free rate
    n_paths : number of MC paths
    n_steps_per_year : time steps per year
    seed : RNG seed

    Returns
    -------
    price : cat bond price (as fraction of notional)
    """
    n_steps = int(bond.maturity * n_steps_per_year)
    dt = bond.maturity / n_steps
    key = jax.random.PRNGKey(seed)

    if isinstance(bond.risk, BetaRisk):
        price = _mc_beta_risk(bond, r, n_paths, n_steps, dt, key)
    elif isinstance(bond.risk, EventSetRisk):
        price = _mc_event_set_risk(bond, r, n_paths, n_steps, dt, key)
    else:
        raise ValueError(f"Unknown risk type: {type(bond.risk)}")

    return price


def _mc_beta_risk(bond, r, n_paths, n_steps, dt, key):
    """MC for beta-distributed cat loss."""
    risk = bond.risk
    k1, k2 = jax.random.split(key)

    # Simulate Poisson arrivals and Beta losses
    N = jax.random.poisson(k1, risk.intensity * dt, (n_paths, n_steps))
    losses = jax.random.beta(k2, risk.alpha, risk.beta, (n_paths, n_steps))

    # Cumulative loss (only when events occur)
    event_losses = losses * (N > 0)
    cum_loss = jnp.cumsum(event_losses, axis=1)

    # Final cumulative loss
    final_loss = cum_loss[:, -1]

    # Principal erosion
    recovery_frac = 1.0 - jnp.clip(
        (final_loss - bond.attachment) / bond.width, 0.0, 1.0)

    # Discount factor
    disc = jnp.exp(-r * bond.maturity)

    # Coupon payments (simplified: at maturity)
    coupon = bond.coupon_rate * bond.maturity

    # Price = E[disc * (recovery * notional + coupon * notional)] / notional
    price = float((disc * (recovery_frac + coupon)).mean())
    return price


def _mc_event_set_risk(bond, r, n_paths, n_steps, dt, key):
    """MC for event-set based cat risk."""
    risk = bond.risk
    n_events = len(risk.event_probs)
    event_probs_per_step = risk.event_probs * dt
    key, subkey = jax.random.split(key)

    # Sample events for each path and step
    U = jax.random.uniform(subkey, (n_paths, n_steps, n_events))
    occurred = (U < event_probs_per_step[None, None, :])
    step_losses = (occurred * risk.event_losses[None, None, :]).sum(axis=2)
    cum_loss = jnp.cumsum(step_losses, axis=1)[:, -1]

    recovery_frac = 1.0 - jnp.clip(
        (cum_loss - bond.attachment) / bond.width, 0.0, 1.0)

    disc = jnp.exp(-r * bond.maturity)
    coupon = bond.coupon_rate * bond.maturity
    price = float((disc * (recovery_frac + coupon)).mean())
    return price
