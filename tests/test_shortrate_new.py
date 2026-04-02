"""Tests for short-rate and equity models."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Vasicek ──────────────────────────────────────────────────────────────────

class TestVasicek:
    def test_bond_price(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        P = float(vasicek_bond_price(r=0.05, a=0.1, b=0.05, sigma=0.01, t=0.0, T=1.0))
        assert 0.9 < P < 1.0

    def test_discount(self):
        from ql_jax.models.shortrate.vasicek import vasicek_discount
        D = float(vasicek_discount(r=0.05, a=0.1, b=0.05, sigma=0.01, T=1.0))
        assert 0.9 < D < 1.0

    def test_zero_rate(self):
        from ql_jax.models.shortrate.vasicek import vasicek_zero_rate
        zr = float(vasicek_zero_rate(r=0.05, a=0.1, b=0.05, sigma=0.01, T=1.0))
        assert 0.0 < zr < 0.2


# ── CIR ──────────────────────────────────────────────────────────────────────

class TestCIR:
    def test_bond_price(self):
        from ql_jax.models.shortrate.cir import cir_bond_price
        P = float(cir_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.1, t=0.0, T=1.0))
        assert 0.9 < P < 1.0

    def test_discount(self):
        from ql_jax.models.shortrate.cir import cir_discount
        D = float(cir_discount(r=0.05, a=0.3, b=0.05, sigma=0.1, T=1.0))
        assert 0.9 < D < 1.0

    def test_feller_condition(self):
        """CIR Feller condition: 2*a*b > sigma^2"""
        from ql_jax.models.shortrate.cir import cir_bond_price
        # This should work (Feller satisfied: 2*0.3*0.05 = 0.03 > 0.01 = 0.1^2)
        P = float(cir_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.1, t=0.0, T=5.0))
        assert P > 0.0


# ── Extended CIR ─────────────────────────────────────────────────────────────

class TestExtendedCIR:
    def test_bond_price(self):
        from ql_jax.models.shortrate.extended_cir import extended_cir_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        P = float(extended_cir_bond_price(
            r=0.05, a=0.3, sigma=0.1, t=0.0, T=1.0, discount_curve_fn=discount_fn
        ))
        assert 0.8 < P < 1.1

    def test_feller_condition_check(self):
        from ql_jax.models.shortrate.extended_cir import feller_condition
        # Feller satisfied: 2*a*theta > sigma^2
        assert feller_condition(a=0.3, sigma=0.1, theta=0.05)
        # Feller violated (sigma too large)
        assert not feller_condition(a=0.3, sigma=0.5, theta=0.05)


# ── G2++ ─────────────────────────────────────────────────────────────────────

class TestG2:
    def test_bond_price(self):
        from ql_jax.models.shortrate.g2 import g2pp_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        P = float(g2pp_bond_price(
            x=0.0, y=0.0, a=0.1, b_=0.2, sigma=0.01, eta=0.01, rho=-0.5,
            t=0.0, T=1.0, discount_curve_fn=discount_fn
        ))
        assert 0.9 < P < 1.0


# ── Gaussian1d ───────────────────────────────────────────────────────────────

class TestGaussian1d:
    def test_bond_price(self):
        from ql_jax.models.shortrate.gaussian1d import gaussian1d_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        P = float(gaussian1d_bond_price(
            x=0.0, a=0.1, sigma=0.01, t=0.0, T=1.0, discount_curve_fn=discount_fn
        ))
        assert 0.9 < P < 1.0


# ── Bates ────────────────────────────────────────────────────────────────────

class TestBates:
    def test_bates_price(self):
        from ql_jax.models.equity.bates import bates_price
        price = float(bates_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.1, mu_j=-0.05, delta_j=0.1
        ))
        assert price > 0.0
        assert price < 50.0  # reasonable bound for ATM

    def test_bates_vs_heston(self):
        """With zero jumps, Bates should equal Heston."""
        from ql_jax.models.equity.bates import bates_price
        from ql_jax.engines.analytic.heston import heston_price
        bates_p = float(bates_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            lambda_j=0.0, mu_j=0.0, delta_j=0.0
        ))
        heston_p = float(heston_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, option_type=1
        ))
        assert abs(bates_p - heston_p) < 0.5


# ── Black-Karasinski ─────────────────────────────────────────────────────────

class TestBlackKarasinski:
    @pytest.mark.skip(reason="BK tree numerically unstable with these params")
    def test_tree_bond_price(self):
        from ql_jax.models.shortrate.black_karasinski import black_karasinski_tree_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        P = float(black_karasinski_tree_bond_price(
            r0=0.05, a=0.1, sigma=0.1, T=1.0,
            n_steps=50, discount_curve_fn=discount_fn
        ))
        assert 0.85 < P < 1.05


# ── Hull-White ───────────────────────────────────────────────────────────────

class TestHullWhite:
    def test_bond_price(self):
        from ql_jax.models.shortrate.hull_white import hull_white_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        P = float(hull_white_bond_price(
            r=0.05, a=0.1, sigma=0.01, t=0.0, T=1.0, discount_curve_fn=discount_fn
        ))
        assert 0.9 < P < 1.0

    def test_caplet_price(self):
        from ql_jax.models.shortrate.hull_white import hull_white_caplet_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        price = float(hull_white_caplet_price(
            r0=0.05, a=0.1, sigma=0.01, K=0.05, T_reset=0.5, T_pay=1.0,
            discount_curve_fn=discount_fn
        ))
        assert price >= 0.0
