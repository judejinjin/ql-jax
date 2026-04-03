"""Extended Ornstein-Uhlenbeck process with time-dependent parameters.

Also includes OU with jumps for energy/commodity modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ExtendedOrnsteinUhlenbeckProcess:
    """OU process with time-dependent parameters.

    dx = speed(t)*(level(t) - x)dt + sigma(t)*dW

    Parameters
    ----------
    x0 : initial value
    speed : mean reversion speed (float or callable(t))
    level : mean reversion level (float or callable(t))
    sigma : volatility (float or callable(t))
    """
    x0: float
    speed: float = 1.0
    level: float = 0.0
    sigma: float = 0.1

    def mean(self, t):
        """E[x(t)] for constant parameters."""
        a = self.speed
        return self.level + (self.x0 - self.level) * jnp.exp(-a * t)

    def variance(self, t):
        """Var[x(t)] for constant parameters."""
        a = self.speed
        return self.sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t))

    def evolve(self, x, dt, dW):
        """Single Euler step."""
        return x + self.speed * (self.level - x) * dt + self.sigma * dW

    def simulate(self, T, n_steps, n_paths, key):
        """Full path simulation."""
        dt = T / n_steps
        sqrt_dt = jnp.sqrt(dt)
        x = jnp.ones(n_paths) * self.x0
        paths = [x]
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            dW = jax.random.normal(subkey, (n_paths,)) * sqrt_dt
            x = self.evolve(x, dt, dW)
            paths.append(x)
        return jnp.stack(paths)


@dataclass(frozen=True)
class ExtOUWithJumpsProcess:
    """Extended OU with compound Poisson jumps.

    dx = speed*(level - x)*dt + sigma*dW + J*dN

    where J ~ N(mu_j, sigma_j^2) and N is Poisson with intensity lambda.

    Parameters
    ----------
    x0 : initial value
    speed, level, sigma : OU parameters
    lam : jump intensity
    mu_j : mean jump size
    sigma_j : jump size volatility
    """
    x0: float
    speed: float = 1.0
    level: float = 0.0
    sigma: float = 0.1
    lam: float = 0.5
    mu_j: float = 0.0
    sigma_j: float = 0.05

    def simulate(self, T, n_steps, n_paths, key):
        dt = T / n_steps
        sqrt_dt = jnp.sqrt(dt)
        x = jnp.ones(n_paths) * self.x0
        paths = [x]
        for step in range(n_steps):
            key, k1, k2, k3 = jax.random.split(key, 4)
            dW = jax.random.normal(k1, (n_paths,)) * sqrt_dt
            # Poisson jumps
            n_jumps = jax.random.poisson(k2, self.lam * dt, (n_paths,))
            jump_sizes = self.mu_j * n_jumps + self.sigma_j * jnp.sqrt(
                jnp.maximum(n_jumps.astype(jnp.float64), 0.0)
            ) * jax.random.normal(k3, (n_paths,))
            x = x + self.speed * (self.level - x) * dt + self.sigma * dW + jump_sizes
            paths.append(x)
        return jnp.stack(paths)
