"""Tests for Phase 6: Interest Rate Models & Derivatives."""

import pytest
import jax
import jax.numpy as jnp

# ===== Short-Rate Models =====

class TestVasicek:
    def test_bond_price_positive(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        P = vasicek_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.01, t=0.0, T=1.0)
        assert 0.0 < float(P) < 1.0

    def test_bond_price_at_zero_maturity(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        P = vasicek_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.01, t=0.0, T=0.0)
        assert abs(float(P) - 1.0) < 1e-10

    def test_discount_consistency(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price, vasicek_discount
        P1 = vasicek_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.01, t=0.0, T=2.0)
        P2 = vasicek_discount(r=0.05, a=0.3, b=0.05, sigma=0.01, T=2.0)
        assert abs(float(P1) - float(P2)) < 1e-12

    def test_zero_rate_positive(self):
        from ql_jax.models.shortrate.vasicek import vasicek_zero_rate
        R = vasicek_zero_rate(r=0.05, a=0.3, b=0.05, sigma=0.01, T=5.0)
        assert float(R) > 0

    def test_bond_price_decreases_with_rate(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        P_low = vasicek_bond_price(r=0.03, a=0.3, b=0.05, sigma=0.01, t=0.0, T=5.0)
        P_high = vasicek_bond_price(r=0.08, a=0.3, b=0.05, sigma=0.01, t=0.0, T=5.0)
        assert float(P_low) > float(P_high)

    def test_ad_through_bond_price(self):
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        grad_fn = jax.grad(lambda r: vasicek_bond_price(r, 0.3, 0.05, 0.01, 0.0, 5.0))
        dr = grad_fn(0.05)
        assert jnp.isfinite(dr)
        assert float(dr) < 0  # bond price decreases with rate

    def test_caplet_positive(self):
        from ql_jax.models.shortrate.vasicek import vasicek_caplet_price
        price = vasicek_caplet_price(r=0.05, a=0.3, b=0.05, sigma=0.01,
                                      K=0.05, T_reset=1.0, T_pay=1.25)
        assert float(price) >= 0


class TestCIR:
    def test_bond_price_positive(self):
        from ql_jax.models.shortrate.cir import cir_bond_price
        P = cir_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.05, t=0.0, T=1.0)
        assert 0.0 < float(P) < 1.0

    def test_feller_condition(self):
        """With Feller condition satisfied, bond price behaves well."""
        from ql_jax.models.shortrate.cir import cir_bond_price
        # 2*a*b = 0.03 > sigma^2 = 0.0025
        P = cir_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.05, t=0.0, T=10.0)
        assert 0.0 < float(P) < 1.0

    def test_zero_maturity(self):
        from ql_jax.models.shortrate.cir import cir_bond_price
        P = cir_bond_price(r=0.05, a=0.3, b=0.05, sigma=0.05, t=0.0, T=0.0)
        assert abs(float(P) - 1.0) < 1e-10


class TestHullWhite:
    def setup_method(self):
        self.a = 0.1
        self.sigma = 0.01
        self.r0 = 0.05
        # Flat discount curve
        self.discount_fn = lambda t: jnp.exp(-0.05 * t)

    def test_bond_price_at_zero(self):
        from ql_jax.models.shortrate.hull_white import hull_white_bond_price
        P = hull_white_bond_price(self.r0, self.a, self.sigma, 0.0, 0.0, self.discount_fn)
        assert abs(float(P) - 1.0) < 1e-6

    def test_bond_price_positive(self):
        from ql_jax.models.shortrate.hull_white import hull_white_bond_price
        P = hull_white_bond_price(self.r0, self.a, self.sigma, 0.0, 5.0, self.discount_fn)
        assert 0.0 < float(P) < 1.0

    def test_bond_price_fits_market_at_t0(self):
        """At t=0, HW should approximately reproduce market discount factors."""
        from ql_jax.models.shortrate.hull_white import hull_white_bond_price
        for T in [1.0, 2.0, 5.0]:
            P_hw = hull_white_bond_price(self.r0, self.a, self.sigma, 0.0, T, self.discount_fn)
            P_mkt = self.discount_fn(T)
            assert abs(float(P_hw) - float(P_mkt)) < 0.01, f"Mismatch at T={T}"

    def test_caplet_positive(self):
        from ql_jax.models.shortrate.hull_white import hull_white_caplet_price
        price = hull_white_caplet_price(self.r0, self.a, self.sigma, 0.05,
                                         1.0, 1.25, self.discount_fn)
        assert float(price) >= 0

    def test_ad_through_caplet(self):
        from ql_jax.models.shortrate.hull_white import hull_white_caplet_price
        df = self.discount_fn
        grad_fn = jax.grad(lambda sig: hull_white_caplet_price(
            0.05, 0.1, sig, 0.05, 1.0, 1.25, df))
        vega = grad_fn(0.01)
        assert jnp.isfinite(vega)
        assert float(vega) > 0  # caplet price increases with vol


# ===== Calibration =====

class TestCalibration:
    def test_bfgs_optimizer(self):
        from ql_jax.math.optimization.bfgs import minimize
        # Rosenbrock
        def rosenbrock(x):
            return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
        result = minimize(rosenbrock, jnp.array([-1.0, 1.0]), max_iter=200)
        assert abs(float(result['x'][0]) - 1.0) < 0.1
        assert abs(float(result['x'][1]) - 1.0) < 0.1

    def test_calibrate_vasicek(self):
        """Calibrate Vasicek model to synthetic bond prices."""
        from ql_jax.models.shortrate.vasicek import vasicek_bond_price
        from ql_jax.models.calibration import calibrate_least_squares

        # True parameters
        r_true, a_true, b_true, sigma_true = 0.05, 0.3, 0.06, 0.015
        maturities = jnp.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        true_prices = jnp.array([
            vasicek_bond_price(r_true, a_true, b_true, sigma_true, 0.0, T)
            for T in maturities
        ])

        def model_fn(params):
            a, b, sig = params
            return jnp.array([
                vasicek_bond_price(r_true, a, b, sig, 0.0, T)
                for T in maturities
            ])

        result = calibrate_least_squares(
            model_fn, true_prices, None,
            initial_params=jnp.array([0.2, 0.04, 0.01]),
            param_bounds=[(0.01, 2.0), (0.0, 0.2), (0.001, 0.1)],
            max_iter=200,
        )
        # Should recover approximately correct parameters
        assert result.fun < 1e-6


# ===== Cap/Floor =====

class TestCapFloor:
    def test_make_cap(self):
        from ql_jax.instruments.capfloor import make_cap
        cap = make_cap(strike=0.05, start=0.0, maturity=2.0, frequency=0.25)
        assert cap.n_periods == 8
        assert cap.is_cap

    def test_make_floor(self):
        from ql_jax.instruments.capfloor import make_floor
        floor = make_floor(strike=0.04, start=0.0, maturity=1.0, frequency=0.25)
        assert floor.n_periods == 4
        assert not floor.is_cap

    def test_black_cap_price_positive(self):
        from ql_jax.instruments.capfloor import make_cap
        from ql_jax.engines.capfloor.black import black_capfloor_price

        cap = make_cap(strike=0.05, start=0.0, maturity=2.0, frequency=0.25)
        discount_fn = lambda t: jnp.exp(-0.04 * t)
        forwards = jnp.full(cap.n_periods, 0.05)  # ATM
        vols = jnp.full(cap.n_periods, 0.2)

        price = black_capfloor_price(cap, discount_fn, forwards, vols)
        assert float(price) > 0

    def test_cap_floor_parity(self):
        """Cap - Floor = Swap (at same strike)."""
        from ql_jax.instruments.capfloor import make_cap, make_floor
        from ql_jax.engines.capfloor.black import black_capfloor_price

        K = 0.05
        cap = make_cap(strike=K, start=0.0, maturity=2.0, frequency=0.25)
        floor = make_floor(strike=K, start=0.0, maturity=2.0, frequency=0.25)

        discount_fn = lambda t: jnp.exp(-0.04 * t)
        forwards = jnp.full(cap.n_periods, 0.06)  # slightly ITM for cap
        vols = jnp.full(cap.n_periods, 0.2)

        cap_price = black_capfloor_price(cap, discount_fn, forwards, vols)
        floor_price = black_capfloor_price(floor, discount_fn, forwards, vols)

        # Cap - Floor = sum of (F - K) * tau * df
        swap_val = 0.0
        for i in range(cap.n_periods):
            tau = float(cap.accrual_fractions[i])
            T_pay = float(cap.payment_dates[i])
            swap_val += (0.06 - K) * tau * float(discount_fn(T_pay))

        assert abs(float(cap_price - floor_price) - swap_val) < 0.001

    def test_ad_through_cap(self):
        from ql_jax.instruments.capfloor import make_cap
        from ql_jax.engines.capfloor.black import black_capfloor_price

        cap = make_cap(strike=0.05, start=0.0, maturity=1.0, frequency=0.25)
        discount_fn = lambda t: jnp.exp(-0.04 * t)
        forwards = jnp.full(cap.n_periods, 0.05)

        def price_fn(vol):
            vols = jnp.full(cap.n_periods, vol)
            return black_capfloor_price(cap, discount_fn, forwards, vols)

        vega = jax.grad(price_fn)(0.2)
        assert jnp.isfinite(vega)
        assert float(vega) > 0


# ===== Swaption =====

class TestSwaption:
    def test_make_swaption(self):
        from ql_jax.instruments.swaption import make_swaption
        swpn = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=0.05)
        assert swpn.n_periods == 10  # 5y / 0.5y
        assert abs(swpn.swap_maturity - 6.0) < 1e-10

    def test_black_swaption_positive(self):
        from ql_jax.instruments.swaption import make_swaption
        from ql_jax.engines.swaption.black import black_swaption_price

        swpn = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=0.05, payer=True)
        discount_fn = lambda t: jnp.exp(-0.04 * t)
        fwd_rate = 0.05
        vol = 0.15

        price = black_swaption_price(swpn, discount_fn, fwd_rate, vol)
        assert float(price) > 0

    def test_payer_receiver_parity(self):
        """Payer - Receiver = PV of forward swap."""
        from ql_jax.instruments.swaption import make_swaption
        from ql_jax.engines.swaption.black import black_swaption_price, par_swap_rate

        discount_fn = lambda t: jnp.exp(-0.04 * t)
        K = 0.05

        payer = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=K, payer=True)
        receiver = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=K, payer=False)

        fwd = par_swap_rate(discount_fn, 1.0, payer.swap_payment_dates, payer.swap_accrual_fractions)
        vol = 0.15

        p_payer = black_swaption_price(payer, discount_fn, fwd, vol)
        p_receiver = black_swaption_price(receiver, discount_fn, fwd, vol)

        # Payer - Receiver = annuity * (F - K)
        annuity = 0.0
        for i in range(payer.n_periods):
            tau_i = float(payer.swap_accrual_fractions[i])
            T_i = float(payer.swap_payment_dates[i])
            annuity += tau_i * float(discount_fn(T_i))

        expected_diff = annuity * (float(fwd) - K)
        assert abs(float(p_payer - p_receiver) - expected_diff) < 0.001

    def test_par_swap_rate(self):
        from ql_jax.instruments.swaption import make_swaption
        from ql_jax.engines.swaption.black import par_swap_rate

        discount_fn = lambda t: jnp.exp(-0.04 * t)
        swpn = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=0.05)
        fwd = par_swap_rate(discount_fn, 1.0, swpn.swap_payment_dates, swpn.swap_accrual_fractions)
        # For flat curve at 4%, par rate should be close to 4%
        assert abs(float(fwd) - 0.04) < 0.005

    def test_ad_through_swaption(self):
        from ql_jax.instruments.swaption import make_swaption
        from ql_jax.engines.swaption.black import black_swaption_price

        swpn = make_swaption(exercise_time=1.0, swap_tenor=5.0, fixed_rate=0.05, payer=True)
        discount_fn = lambda t: jnp.exp(-0.04 * t)

        def price_fn(vol):
            return black_swaption_price(swpn, discount_fn, 0.05, vol)

        vega = jax.grad(price_fn)(0.15)
        assert jnp.isfinite(vega)
        assert float(vega) > 0


# ===== GARCH =====

class TestGARCH:
    def test_garch_loglikelihood(self):
        from ql_jax.models.volatility.garch import garch11_loglikelihood
        returns = jax.random.normal(jax.random.PRNGKey(42), (200,)) * 0.01
        params = jnp.array([1e-6, 0.05, 0.9])
        ll = garch11_loglikelihood(params, returns)
        assert jnp.isfinite(ll)

    def test_garch_forecast(self):
        from ql_jax.models.volatility.garch import garch11_forecast
        returns = jax.random.normal(jax.random.PRNGKey(42), (200,)) * 0.01
        params = jnp.array([1e-6, 0.05, 0.9])
        forecasts = garch11_forecast(params, returns, n_ahead=5)
        assert forecasts.shape == (5,)
        assert jnp.all(forecasts > 0)

    def test_garch_ad(self):
        from ql_jax.models.volatility.garch import garch11_loglikelihood
        returns = jax.random.normal(jax.random.PRNGKey(42), (100,)) * 0.01

        def ll_fn(params):
            return garch11_loglikelihood(params, returns)

        grad = jax.grad(ll_fn)(jnp.array([1e-6, 0.05, 0.9]))
        assert jnp.all(jnp.isfinite(grad))

    def test_realized_vol(self):
        from ql_jax.models.volatility.garch import realized_volatility
        returns = jax.random.normal(jax.random.PRNGKey(42), (252,)) * 0.01
        vol = realized_volatility(returns)
        assert 0.1 < float(vol) < 0.2  # ~16% annualized for 1% daily


# ===== Heston Calibration =====

class TestHestonCalibration:
    def test_heston_model_prices(self):
        from ql_jax.models.equity.heston import heston_model_prices
        params = jnp.array([0.04, 2.0, 0.04, 0.5, -0.7])
        strikes = jnp.array([90.0, 100.0, 110.0])
        maturities = jnp.array([0.5, 0.5, 0.5])
        from ql_jax._util.types import OptionType
        types = jnp.array([OptionType.Call, OptionType.Call, OptionType.Call])
        prices = heston_model_prices(params, 100.0, 0.05, 0.02, strikes, maturities, types)
        assert prices.shape == (3,)
        assert jnp.all(prices > 0)
