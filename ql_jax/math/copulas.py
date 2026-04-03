"""Copula implementations.

Bivariate copulas used in credit correlation models, CMS spread pricing,
and other multivariate dependence modeling.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
import math


# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------

def gaussian_copula(u: float, v: float, rho: float) -> float:
    """Bivariate Gaussian copula C(u,v; rho).

    C(u,v) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)
    where Phi_2 is the standard bivariate normal CDF.
    """
    x = jax_norm.ppf(jnp.clip(u, 1e-15, 1 - 1e-15))
    y = jax_norm.ppf(jnp.clip(v, 1e-15, 1 - 1e-15))
    from ql_jax.math.distributions.bivariate import bivariate_normal_cdf
    return float(bivariate_normal_cdf(x, y, rho))


# ---------------------------------------------------------------------------
# Clayton copula
# ---------------------------------------------------------------------------

def clayton_copula(u: float, v: float, theta: float) -> float:
    r"""Clayton copula: C(u,v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}.

    theta > 0 gives lower tail dependence.
    """
    if theta <= 0:
        raise ValueError("Clayton copula requires theta > 0")
    val = jnp.power(u, -theta) + jnp.power(v, -theta) - 1.0
    val = jnp.maximum(val, 0.0)
    return float(jnp.power(val, -1.0 / theta))


# ---------------------------------------------------------------------------
# Frank copula
# ---------------------------------------------------------------------------

def frank_copula(u: float, v: float, theta: float) -> float:
    r"""Frank copula: symmetric dependence.

    C(u,v) = -\frac{1}{\theta} \ln\left(1 + \frac{(e^{-\theta u} - 1)(e^{-\theta v} - 1)}{e^{-\theta} - 1}\right)
    """
    if abs(theta) < 1e-10:
        return float(u * v)  # independence limit
    e_t = jnp.exp(-theta)
    num = (jnp.exp(-theta * u) - 1.0) * (jnp.exp(-theta * v) - 1.0)
    return float(-jnp.log1p(num / (e_t - 1.0)) / theta)


# ---------------------------------------------------------------------------
# Gumbel copula
# ---------------------------------------------------------------------------

def gumbel_copula(u: float, v: float, theta: float) -> float:
    r"""Gumbel copula: upper tail dependence.

    C(u,v) = \exp\left(-((-\ln u)^\theta + (-\ln v)^\theta)^{1/\theta}\right)
    theta >= 1; theta=1 gives independence.
    """
    if theta < 1.0:
        raise ValueError("Gumbel copula requires theta >= 1")
    lu = jnp.power(-jnp.log(jnp.clip(u, 1e-15, 1.0)), theta)
    lv = jnp.power(-jnp.log(jnp.clip(v, 1e-15, 1.0)), theta)
    return float(jnp.exp(-jnp.power(lu + lv, 1.0 / theta)))


# ---------------------------------------------------------------------------
# Independent copula
# ---------------------------------------------------------------------------

def independent_copula(u: float, v: float) -> float:
    """Product (independence) copula: C(u,v) = u*v."""
    return float(u * v)


# ---------------------------------------------------------------------------
# Min / Max copulas (Fréchet–Hoeffding bounds)
# ---------------------------------------------------------------------------

def min_copula(u: float, v: float) -> float:
    """Upper Fréchet–Hoeffding bound: C(u,v) = min(u,v)."""
    return float(jnp.minimum(u, v))


def max_copula(u: float, v: float) -> float:
    """Lower Fréchet–Hoeffding bound: C(u,v) = max(u+v-1, 0)."""
    return float(jnp.maximum(u + v - 1.0, 0.0))


# ---------------------------------------------------------------------------
# Plackett copula
# ---------------------------------------------------------------------------

def plackett_copula(u: float, v: float, theta: float) -> float:
    r"""Plackett copula.

    theta > 0, theta != 1. theta=1 gives independence.
    """
    if abs(theta - 1.0) < 1e-10:
        return float(u * v)
    s = u + v
    eta = theta - 1.0
    inner = (1.0 + eta * s) ** 2 - 4.0 * theta * eta * u * v
    inner = jnp.maximum(inner, 0.0)
    return float((1.0 + eta * s - jnp.sqrt(inner)) / (2.0 * eta))


# ---------------------------------------------------------------------------
# Farlie-Gumbel-Morgenstern copula
# ---------------------------------------------------------------------------

def fgm_copula(u: float, v: float, theta: float) -> float:
    r"""Farlie-Gumbel-Morgenstern copula.

    C(u,v) = uv + \theta uv(1-u)(1-v)
    |theta| <= 1.
    """
    return float(u * v + theta * u * v * (1.0 - u) * (1.0 - v))


# ---------------------------------------------------------------------------
# Galambos copula
# ---------------------------------------------------------------------------

def galambos_copula(u: float, v: float, theta: float) -> float:
    r"""Galambos copula (extreme-value copula).

    C(u,v) = uv \exp\left(((-\ln u)^{-\theta} + (-\ln v)^{-\theta})^{-1/\theta}\right)
    theta > 0.
    """
    if theta <= 0:
        raise ValueError("Galambos copula requires theta > 0")
    lu = -jnp.log(jnp.clip(u, 1e-15, 1.0))
    lv = -jnp.log(jnp.clip(v, 1e-15, 1.0))
    A = jnp.power(jnp.power(lu, -theta) + jnp.power(lv, -theta), -1.0 / theta)
    return float(u * v * jnp.exp(A))


# ---------------------------------------------------------------------------
# Husler-Reiss copula
# ---------------------------------------------------------------------------

def husler_reiss_copula(u: float, v: float, theta: float) -> float:
    r"""Husler-Reiss copula (extreme-value copula).

    theta > 0; theta -> inf gives comonotonicity, theta -> 0 gives independence.
    """
    if theta <= 0:
        raise ValueError("Husler-Reiss copula requires theta > 0")
    lu = -jnp.log(jnp.clip(u, 1e-15, 1.0))
    lv = -jnp.log(jnp.clip(v, 1e-15, 1.0))
    r = jnp.where(lv > 0, lu / lv, 1.0)
    z1 = 1.0 / theta + 0.5 * theta * jnp.log(r)
    z2 = 1.0 / theta - 0.5 * theta * jnp.log(r)
    from jax.scipy.stats import norm
    return float(jnp.exp(-lu * norm.cdf(z1) - lv * norm.cdf(z2)))


# ---------------------------------------------------------------------------
# Ali-Mikhail-Haq copula
# ---------------------------------------------------------------------------

def ali_mikhail_haq_copula(u: float, v: float, theta: float) -> float:
    r"""Ali-Mikhail-Haq copula.

    C(u,v) = uv / (1 - theta(1-u)(1-v))
    -1 <= theta <= 1
    """
    denom = 1.0 - theta * (1.0 - u) * (1.0 - v)
    return float(u * v / denom)


# ---------------------------------------------------------------------------
# Marshall-Olkin copula
# ---------------------------------------------------------------------------

def marshall_olkin_copula(u: float, v: float, alpha: float, beta: float) -> float:
    r"""Marshall-Olkin copula.

    C(u,v) = min(u^{1-alpha} v, u v^{1-beta})
    0 <= alpha, beta <= 1
    """
    a = jnp.power(u, 1.0 - alpha) * v
    b = u * jnp.power(v, 1.0 - beta)
    return float(jnp.minimum(a, b))
