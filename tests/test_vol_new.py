"""Tests for volatility surfaces and related term structures."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date


# ── Local Vol Surface ────────────────────────────────────────────────────────

class TestLocalVol:
    def test_dupire_local_vol(self):
        from ql_jax.termstructures.volatility.local_vol_surface import dupire_local_vol
        # Simple flat implied vol surface
        implied_vol_fn = lambda T, K: 0.2
        lv = float(dupire_local_vol(T=1.0, K=100.0, S=100.0, r=0.05, q=0.0, implied_vol_fn=implied_vol_fn))
        assert lv > 0.0
        assert lv < 1.0


# ── SABR Functions ───────────────────────────────────────────────────────────

class TestSABR:
    def test_sabr_vol_atm(self):
        from ql_jax.termstructures.volatility.sabr_functions import sabr_vol
        # With beta=0.5, alpha=0.2 on F=K=100, the SABR vol may be small
        vol = float(sabr_vol(
            strike=100.0, forward=100.0, t=1.0,
            alpha=0.2, beta=0.5, rho=-0.3, nu=0.4
        ))
        assert vol > 0.0
        assert vol < 1.0

    def test_sabr_smile(self):
        from ql_jax.termstructures.volatility.sabr_functions import sabr_vol
        vol_atm = float(sabr_vol(strike=100.0, forward=100.0, t=1.0,
                                  alpha=0.2, beta=0.5, rho=-0.5, nu=0.4))
        vol_otm = float(sabr_vol(strike=80.0, forward=100.0, t=1.0,
                                  alpha=0.2, beta=0.5, rho=-0.5, nu=0.4))
        # Both should be positive
        assert vol_atm > 0.0
        assert vol_otm > 0.0

    def test_sabr_normal_vol(self):
        from ql_jax.termstructures.volatility.sabr_functions import sabr_vol_normal
        vol = float(sabr_vol_normal(
            strike=100.0, forward=100.0, t=1.0,
            alpha=0.2, beta=0.5, rho=-0.3, nu=0.4
        ))
        assert vol > 0.0


# ── Smile Section ────────────────────────────────────────────────────────────

class TestSmileSection:
    def test_flat_smile(self):
        from ql_jax.termstructures.volatility.smile_section import FlatSmileSection
        smile = FlatSmileSection(expiry_time=1.0, vol=0.2, forward=100.0)
        vol = float(smile.volatility(90.0))
        assert abs(vol - 0.2) < 1e-10

    def test_sabr_smile_section(self):
        from ql_jax.termstructures.volatility.smile_section import SABRSmileSection
        smile = SABRSmileSection(
            expiry_time=1.0, forward=100.0,
            alpha=0.2, beta=0.5, rho=-0.3, nu=0.4
        )
        vol = float(smile.volatility(100.0))
        assert vol > 0.0


# ── Black Vol Term Structures ────────────────────────────────────────────────

class TestBlackVol:
    def test_constant_vol(self):
        from ql_jax.termstructures.volatility.black_vol import BlackConstantVol
        ref_date = Date(1, 1, 2024)
        ts = BlackConstantVol(reference_date=ref_date, vol=0.25)
        vol = float(ts.black_vol(1.0, 100.0))
        assert abs(vol - 0.25) < 1e-10


# ── Cap/Floor Vol ────────────────────────────────────────────────────────────

class TestCapFloorVol:
    def test_constant_capfloor_vol(self):
        from ql_jax.termstructures.volatility.capfloor_vol import ConstantCapFloorTermVol
        ref_date = Date(1, 1, 2024)
        ts = ConstantCapFloorTermVol(reference_date=ref_date, vol=0.15)
        vol = float(ts.volatility(1.0))
        assert abs(vol - 0.15) < 1e-10


# ── Swaption Vol ─────────────────────────────────────────────────────────────

class TestSwaptionVol:
    def test_constant_swaption_vol(self):
        from ql_jax.termstructures.volatility.swaption_vol import SwaptionConstantVol
        ref_date = Date(1, 1, 2024)
        ts = SwaptionConstantVol(reference_date=ref_date, vol=0.12)
        vol = float(ts.volatility(1.0, 5.0))
        assert abs(vol - 0.12) < 1e-10


# ── Optionlet Stripper ───────────────────────────────────────────────────────

class TestOptionletStripper:
    def test_strip_basic(self):
        from ql_jax.termstructures.volatility.optionlet_stripper import strip_optionlet_vols
        cap_maturities = jnp.array([1.0, 2.0, 3.0])
        cap_vols = jnp.array([0.20, 0.19, 0.18])
        forward_rates = jnp.array([0.05, 0.05, 0.05])
        accruals = jnp.array([1.0, 1.0, 1.0])
        discount_factors = jnp.exp(-0.05 * cap_maturities)
        opt_vols = strip_optionlet_vols(
            cap_maturities=cap_maturities, cap_vols=cap_vols,
            forward_rates=forward_rates, accruals=accruals,
            discount_factors=discount_factors
        )
        assert len(opt_vols) == 3
        assert all(float(v) > 0.0 for v in opt_vols)
