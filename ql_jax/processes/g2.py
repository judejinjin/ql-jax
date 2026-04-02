"""G2++ two-factor short-rate process.

r(t) = x(t) + y(t) + phi(t)
dx = -a*x dt + sigma dW_1
dy = -b*y dt + eta dW_2
<dW_1, dW_2> = rho dt

phi(t) chosen to match initial term structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class G2Process:
    """G2++ two-factor Gaussian short-rate process.

    Parameters
    ----------
    a : mean reversion speed for factor x
    sigma : volatility for factor x
    b : mean reversion speed for factor y
    eta : volatility for factor y
    rho : correlation between the two factors
    """
    a: float
    sigma: float
    b: float
    eta: float
    rho: float

    def drift(self, t, state):
        """Drift of state (x, y)."""
        x, y = state
        return jnp.array([-self.a * x, -self.b * y])

    def diffusion(self, t, state):
        """Diffusion matrix (2x2) with correlation."""
        return jnp.array([
            [self.sigma, 0.0],
            [self.eta * self.rho, self.eta * jnp.sqrt(1.0 - self.rho**2)]
        ])

    def expectation(self, t0, state, dt):
        x0, y0 = state
        ex = x0 * jnp.exp(-self.a * dt)
        ey = y0 * jnp.exp(-self.b * dt)
        return jnp.array([ex, ey])

    def covariance(self, t0, state, dt):
        """Covariance matrix of (x,y) over dt."""
        ea = jnp.exp(-self.a * dt)
        eb = jnp.exp(-self.b * dt)

        var_x = self.sigma**2 / (2.0 * self.a) * (1.0 - ea**2)
        var_y = self.eta**2 / (2.0 * self.b) * (1.0 - eb**2)
        cov_xy = (self.rho * self.sigma * self.eta / (self.a + self.b)
                  * (1.0 - ea * eb))

        return jnp.array([[var_x, cov_xy], [cov_xy, var_y]])

    def evolve(self, t0, state, dt, dw):
        """Exact step for state (x, y).

        Parameters
        ----------
        state : (x, y)
        dw : (dw1, dw2) independent standard normals * sqrt(dt)

        Returns
        -------
        (x_new, y_new)
        """
        x0, y0 = state
        dw1, dw2 = dw

        # Exact mean
        mean = self.expectation(t0, state, dt)

        # Exact covariance
        cov = self.covariance(t0, state, dt)

        # Cholesky decomposition for correlated draws
        L00 = jnp.sqrt(cov[0, 0])
        L10 = cov[1, 0] / L00
        L11 = jnp.sqrt(jnp.maximum(cov[1, 1] - L10**2, 0.0))

        z1 = dw1 / jnp.sqrt(dt)
        z2 = dw2 / jnp.sqrt(dt)

        x_new = mean[0] + L00 * z1
        y_new = mean[1] + L10 * z1 + L11 * z2

        return (x_new, y_new)

    def V(self, t):
        """Deterministic function V(t) for bond pricing.

        V(t) = sigma^2/a^2 * [t + 2/a*exp(-a*t) - 1/(2a)*exp(-2at) - 3/(2a)]
             + eta^2/b^2 * [t + 2/b*exp(-b*t) - 1/(2b)*exp(-2bt) - 3/(2b)]
             + 2*rho*sigma*eta/(a*b) * [t + (exp(-at)-1)/a + (exp(-bt)-1)/b
                                         - (exp(-(a+b)t)-1)/(a+b)]
        """
        a, b = self.a, self.b
        s, e = self.sigma, self.eta

        ea = jnp.exp(-a * t)
        eb = jnp.exp(-b * t)

        v = (s**2 / a**2 * (t + 2.0 / a * ea - 1.0 / (2.0 * a) * ea**2 - 3.0 / (2.0 * a))
             + e**2 / b**2 * (t + 2.0 / b * eb - 1.0 / (2.0 * b) * eb**2 - 3.0 / (2.0 * b))
             + 2.0 * self.rho * s * e / (a * b)
             * (t + (ea - 1.0) / a + (eb - 1.0) / b - (jnp.exp(-(a + b) * t) - 1.0) / (a + b)))
        return v
