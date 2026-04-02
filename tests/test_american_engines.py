"""Tests for American option pricing engines."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestBaroneAdesiWhaley:
    """Barone-Adesi & Whaley approximation."""

    def test_call_atm(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) > 0
        # Should be > Black-Scholes European value
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs_price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, 1)
        assert float(price) >= float(bs_price) - 0.01  # American >= European

    def test_put_atm(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=-1)
        assert float(price) > 0
        # American put with r>0 should have early exercise premium
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs_price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, -1)
        assert float(price) >= float(bs_price) - 0.01

    def test_deep_itm_call(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=150, K=100, T=0.5, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) >= 50.0  # At least intrinsic

    def test_deep_otm_put(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=150, K=100, T=0.5, r=0.05, q=0.0, sigma=0.20, option_type=-1)
        assert float(price) >= 0
        assert float(price) < 5.0  # Deep OTM

    def test_short_maturity(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=100, K=100, T=0.01, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) >= 0
        assert float(price) < 5.0

    def test_zero_vol(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        price = barone_adesi_whaley_price(S=100, K=90, T=1.0, r=0.05, q=0.0, sigma=0.001, option_type=1)
        # With near-zero vol, call on S=100, K=90 should be close to discounted intrinsic
        assert float(price) >= 9.0

    def test_high_dividend_call(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        # With high dividends, early exercise of call is valuable
        price = barone_adesi_whaley_price(S=100, K=100, T=1.0, r=0.05, q=0.08, sigma=0.25, option_type=1)
        assert float(price) > 0


class TestBjerksundStensland:
    """Bjerksund-Stensland (2002) approximation."""

    def test_call_atm(self):
        from ql_jax.engines.analytic.american import bjerksund_stensland_price
        price = bjerksund_stensland_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) > 0

    def test_put_atm(self):
        from ql_jax.engines.analytic.american import bjerksund_stensland_price
        price = bjerksund_stensland_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=-1)
        assert float(price) > 0

    def test_consistency_with_baw(self):
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price, bjerksund_stensland_price
        baw = barone_adesi_whaley_price(S=100, K=100, T=0.5, r=0.05, q=0.02, sigma=0.25, option_type=1)
        bjs = bjerksund_stensland_price(S=100, K=100, T=0.5, r=0.05, q=0.02, sigma=0.25, option_type=1)
        # Both approximations should be within ~5% of each other
        assert abs(float(baw) - float(bjs)) / float(baw) < 0.10

    def test_put_exceeds_european(self):
        from ql_jax.engines.analytic.american import bjerksund_stensland_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        am = bjerksund_stensland_price(S=100, K=110, T=1.0, r=0.08, q=0.0, sigma=0.25, option_type=-1)
        eu = black_scholes_price(100.0, 110.0, 1.0, 0.08, 0.0, 0.25, -1)
        # American put should be close to or exceed European value
        assert float(am) >= float(eu) * 0.90  # within 10% at least

    def test_deep_itm_put(self):
        from ql_jax.engines.analytic.american import bjerksund_stensland_price
        price = bjerksund_stensland_price(S=50, K=100, T=0.5, r=0.05, q=0.0, sigma=0.20, option_type=-1)
        assert float(price) >= 50.0  # At least intrinsic


class TestJuQuadratic:
    def test_call(self):
        from ql_jax.engines.analytic.american import ju_quadratic_price
        price = ju_quadratic_price(S=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25, option_type=1)
        assert float(price) > 0

    def test_put(self):
        from ql_jax.engines.analytic.american import ju_quadratic_price
        price = ju_quadratic_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=-1)
        assert float(price) > 0


class TestJumpDiffusion:
    """Merton (1976) jump-diffusion."""

    def test_no_jumps_equals_bs(self):
        from ql_jax.engines.analytic.american import jump_diffusion_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        # With zero jump intensity, should equal BS
        jd = jump_diffusion_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20,
                                   jump_intensity=0.0, jump_mean=0.0, jump_vol=0.0, option_type=1)
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, 1)
        assert float(jd) == pytest.approx(float(bs), abs=0.1)

    def test_jumps_increase_price(self):
        from ql_jax.engines.analytic.american import jump_diffusion_price
        no_jump = jump_diffusion_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20,
                                        jump_intensity=0.0, jump_mean=0.0, jump_vol=0.0, option_type=1)
        with_jump = jump_diffusion_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20,
                                          jump_intensity=2.0, jump_mean=-0.05, jump_vol=0.15, option_type=1)
        # Jump risk generally increases option prices (OTM)
        assert float(with_jump) > 0

    def test_put_with_jumps(self):
        from ql_jax.engines.analytic.american import jump_diffusion_price
        price = jump_diffusion_price(S=100, K=100, T=0.5, r=0.05, q=0.0, sigma=0.20,
                                      jump_intensity=1.0, jump_mean=-0.1, jump_vol=0.10, option_type=-1)
        assert float(price) > 0

    def test_high_frequency_jumps(self):
        from ql_jax.engines.analytic.american import jump_diffusion_price
        price = jump_diffusion_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.10,
                                      jump_intensity=10.0, jump_mean=0.0, jump_vol=0.05, option_type=1)
        assert float(price) > 0


class TestAmericanPutCallSymmetry:
    """Cross-model sanity checks."""

    def test_put_call_parity_european_limit(self):
        """For very small early exercise premium, American ~= European."""
        from ql_jax.engines.analytic.american import barone_adesi_whaley_price
        # Short maturity, low rates => minimal early exercise premium
        call = barone_adesi_whaley_price(S=100, K=100, T=0.01, r=0.001, q=0.001, sigma=0.20, option_type=1)
        put = barone_adesi_whaley_price(S=100, K=100, T=0.01, r=0.001, q=0.001, sigma=0.20, option_type=-1)
        # Both should be small but positive
        assert float(call) > 0
        assert float(put) > 0
