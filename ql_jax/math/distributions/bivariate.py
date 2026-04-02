"""Bivariate normal and bivariate Student-t distributions."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def bivariate_normal_cdf(x, y, rho):
    """Bivariate normal CDF using Drezner-Wesolowsky approximation.

    P(X <= x, Y <= y) where (X,Y) ~ BVN(0, 0, 1, 1, rho)
    """
    # Identity: M(x, y, rho) = Phi(x) - M(x, -y, -rho) when rho < 0
    if_neg_rho = rho < 0

    # When rho < 0, compute M(x, -y, |rho|), then result = Phi(x) - that
    y_adj = jnp.where(if_neg_rho, -y, y)
    rho_adj = jnp.where(if_neg_rho, -rho, rho)

    a = jnp.array([0.3253030, 0.4211071, 0.1334425, 0.006374323])
    b = jnp.array([0.1337764, 0.6243247, 1.3425378, 2.2626645])

    h = -x
    k = -y_adj

    bvn = 0.0

    # Low correlation path — Gauss-Legendre
    hs = (h * h + k * k) / 2.0
    asr = jnp.arcsin(rho_adj)

    for i in range(4):
        for sign in [-1.0, 1.0]:
            sn = jnp.sin(asr * (sign * b[i] + 1.0) / 2.0)
            bvn += a[i] * jnp.exp((sn * h * k - hs) / (1.0 - sn * sn))

    bvn = bvn * asr / (4.0 * jnp.pi)
    bvn += norm.cdf(-h) * norm.cdf(-k)

    # Correction for negative rho: M(x, y, rho) = Phi(x) - M(x, -y, |rho|)
    bvn = jnp.where(if_neg_rho, norm.cdf(x) - bvn, bvn)
    bvn = jnp.clip(bvn, 0.0, 1.0)

    return bvn


def bivariate_normal_pdf(x, y, rho):
    """Bivariate normal PDF."""
    det = 1.0 - rho**2
    z = (x**2 - 2.0 * rho * x * y + y**2) / det
    return jnp.exp(-0.5 * z) / (2.0 * jnp.pi * jnp.sqrt(det))
