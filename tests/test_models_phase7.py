"""Tests for Phase 7 model gaps: vol estimators, Heston SLV, market models, Brownian generators."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

class TestConstantEstimator:
    def test_known_vol(self):
        from ql_jax.models.volatility.estimators import constant_estimator
        # Generate prices from GBM with known vol
        key = jax.random.PRNGKey(42)
        n = 252
        sigma_true = 0.20
        log_returns = sigma_true / jnp.sqrt(252) * jax.random.normal(key, (n,))
        prices = 100.0 * jnp.exp(jnp.cumsum(log_returns))
        vol = constant_estimator(prices, window=252)
        assert float(vol) == pytest.approx(sigma_true, rel=0.3)

    def test_positive_vol(self):
        from ql_jax.models.volatility.estimators import constant_estimator
        prices = jnp.array([100, 101, 99, 102, 98, 103, 97, 104], dtype=jnp.float64)
        vol = constant_estimator(prices, window=8)
        assert float(vol) > 0


class TestGarmanKlassEstimator:
    def test_positive_vol(self):
        from ql_jax.models.volatility.estimators import garman_klass_estimator
        n = 20
        opens = jnp.full(n, 100.0)
        highs = jnp.full(n, 102.0)
        lows = jnp.full(n, 98.0)
        closes = jnp.full(n, 100.5)
        vol = garman_klass_estimator(opens, highs, lows, closes, window=20)
        assert float(vol) > 0

    def test_zero_range(self):
        from ql_jax.models.volatility.estimators import garman_klass_estimator
        n = 10
        prices = jnp.full(n, 100.0)
        vol = garman_klass_estimator(prices, prices, prices, prices, window=10)
        assert float(vol) >= 0


class TestSimpleLocalEstimator:
    def test_output_shape(self):
        from ql_jax.models.volatility.estimators import simple_local_estimator
        prices = jnp.linspace(90, 110, 50)
        strikes = jnp.array([95.0, 100.0, 105.0])
        result = simple_local_estimator(prices, strikes, window=20)
        assert result.shape == (3,)
        assert jnp.all(result >= 0)


# ---------------------------------------------------------------------------
# Brownian generators
# ---------------------------------------------------------------------------

class TestPseudoRandomBrownian:
    def test_shape(self):
        from ql_jax.models.marketmodels.brownian_generators import pseudo_random_brownian
        key = jax.random.PRNGKey(0)
        w = pseudo_random_brownian(key, n_paths=100, n_steps=50, n_factors=3)
        assert w.shape == (100, 50, 3)

    def test_zero_mean(self):
        from ql_jax.models.marketmodels.brownian_generators import pseudo_random_brownian
        key = jax.random.PRNGKey(1)
        w = pseudo_random_brownian(key, n_paths=50000, n_steps=10, n_factors=1)
        mean = jnp.mean(w)
        assert float(mean) == pytest.approx(0.0, abs=0.05)


class TestSobolBrownianBridge:
    def test_shape(self):
        from ql_jax.models.marketmodels.brownian_generators import sobol_brownian_bridge
        w = sobol_brownian_bridge(n_paths=64, n_steps=16, n_factors=2)
        assert w.shape == (64, 16, 2)


# ---------------------------------------------------------------------------
# Market model accounting
# ---------------------------------------------------------------------------

class TestAccountingEngine:
    def test_swap_npv(self):
        from ql_jax.models.marketmodels.accounting import (
            MarketModelProduct, accounting_engine,
        )
        n_rates = 3
        evol_times = jnp.array([1.0, 2.0, 3.0])
        cf_times = jnp.array([1.0, 2.0, 3.0])

        def cashflow_fn(fwd_rates, t_idx):
            # Fixed-float swap: receive fixed 5%, pay floating; per-path
            return 0.05 - fwd_rates[:, t_idx]

        product = MarketModelProduct(
            n_rates=n_rates,
            evolution_times=evol_times,
            cashflow_times=cf_times,
            cashflow_fn=cashflow_fn,
        )
        init_rates = jnp.array([0.04, 0.045, 0.05])
        taus = jnp.array([1.0, 1.0, 1.0])
        vols = jnp.array([0.20, 0.18, 0.15])
        corr = jnp.eye(n_rates)

        result = accounting_engine(product, init_rates, taus, vols, corr, n_paths=5000, key=jax.random.PRNGKey(42))
        assert hasattr(result, 'npv_mean')
        assert hasattr(result, 'npv_std')
        assert result.npv_std > 0


# ---------------------------------------------------------------------------
# Heston SLV (smoke test)
# ---------------------------------------------------------------------------

class TestHestonSLVMC:
    def test_smoke(self):
        from ql_jax.models.equity.heston_slv_mc import heston_slv_mc_calibrate
        # Simple constant local vol
        market_local_vol_fn = lambda t, S: 0.20
        result = heston_slv_mc_calibrate(
            spot=100.0, r=0.05, q=0.0,
            v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7,
            market_local_vol_fn=market_local_vol_fn, T=0.5,
            n_steps=20, n_paths=2000, key=jax.random.PRNGKey(0),
        )
        assert 'leverage_grid' in result
        assert 'time_grid' in result
        assert 'spot_grid' in result
