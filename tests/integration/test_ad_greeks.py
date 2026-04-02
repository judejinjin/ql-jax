"""Integration test: AD Greeks verification.

Verify that automatic differentiation Greeks match finite-difference bumps
across all pricing engines.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


def _fd_deriv(f, x, eps=1e-5):
    """Central finite difference."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class TestADGreeks:
    """AD Greeks vs finite difference for each engine."""

    def test_bsm_delta(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        f = lambda s: black_scholes_price(s, 100.0, 1.0, 0.05, 0.0, 0.20, 1)
        ad = float(jax.grad(f)(jnp.float64(100.0)))
        fd = float(_fd_deriv(f, jnp.float64(100.0)))
        assert abs(ad - fd) < 1e-6

    def test_bsm_gamma(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        delta_fn = jax.grad(
            lambda s: black_scholes_price(s, 100.0, 1.0, 0.05, 0.0, 0.20, 1)
        )
        gamma_ad = float(jax.grad(delta_fn)(jnp.float64(100.0)))
        gamma_fd = float(_fd_deriv(delta_fn, jnp.float64(100.0), eps=0.01))
        assert abs(gamma_ad - gamma_fd) < 1e-4

    def test_bsm_vega(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        f = lambda sig: black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, sig, 1)
        ad = float(jax.grad(f)(jnp.float64(0.20)))
        fd = float(_fd_deriv(f, jnp.float64(0.20), eps=1e-6))
        assert abs(ad - fd) < 1e-4

    def test_bsm_rho(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        f = lambda r: black_scholes_price(100.0, 100.0, 1.0, r, 0.0, 0.20, 1)
        ad = float(jax.grad(f)(jnp.float64(0.05)))
        fd = float(_fd_deriv(f, jnp.float64(0.05)))
        assert abs(ad - fd) < 1e-5

    def test_bsm_theta(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        f = lambda T: black_scholes_price(100.0, 100.0, T, 0.05, 0.0, 0.20, 1)
        ad = float(jax.grad(f)(jnp.float64(1.0)))
        fd = float(_fd_deriv(f, jnp.float64(1.0), eps=1e-6))
        assert abs(ad - fd) < 1e-4

    def test_heston_delta(self):
        from ql_jax.engines.analytic.heston import heston_price

        f = lambda s: heston_price(s, 100.0, 1.0, 0.05, 0.0, 0.04, 1.0, 0.04, 0.3, -0.7, 1)
        ad = float(jax.grad(f)(jnp.float64(100.0)))
        fd = float(_fd_deriv(f, jnp.float64(100.0), eps=0.01))
        assert abs(ad - fd) < 1e-3

    def test_cds_spread_sensitivity(self):
        """AD through CDS fair spread w.r.t. hazard rate."""
        from ql_jax.instruments.cds import make_cds
        from ql_jax.engines.credit.midpoint import midpoint_cds_npv

        cds = make_cds(1e6, 0.01, 5.0)

        def npv_fn(h):
            return midpoint_cds_npv(
                cds,
                lambda t: jnp.exp(-0.05 * t),
                lambda t: jnp.exp(-h * t),
            )

        ad = float(jax.grad(npv_fn)(jnp.float64(0.02)))
        fd = float(_fd_deriv(npv_fn, jnp.float64(0.02), eps=1e-6))
        assert abs(ad - fd) < 1.0  # within $1 on $1M

    def test_inflation_cap_sensitivity(self):
        """AD through inflation cap w.r.t. forward rate."""
        from ql_jax.engines.inflation.capfloor import (
            InflationCapFloor, black_yoy_capfloor_price,
        )

        dates = jnp.array([1.0, 2.0, 3.0])
        cap = InflationCapFloor('cap', 0.02, 1e6, dates)
        vols = jnp.array([0.01, 0.01, 0.01])

        f = lambda fwd: black_yoy_capfloor_price(
            cap, lambda t: jnp.exp(-0.03 * t), jnp.ones(3) * fwd, vols
        )

        ad = float(jax.grad(f)(jnp.float64(0.025)))
        fd = float(_fd_deriv(f, jnp.float64(0.025), eps=1e-6))
        assert abs(ad - fd) < 1.0
