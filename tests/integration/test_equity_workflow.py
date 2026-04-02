"""Integration test: Equity option pricing workflow.

Build market → price options → compute Greeks → verify consistency across engines.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestEquityWorkflow:
    """End-to-end equity option pricing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.q = 0.02
        self.sigma = 0.20

    def test_all_engines_agree_european_call(self):
        """BSM, binomial, FD all agree on European call price."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        from ql_jax.engines.lattice.binomial import binomial_price
        from ql_jax.engines.fd.black_scholes import fd_black_scholes_price

        bsm = float(black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                          self.sigma, 1))
        tree = float(binomial_price(self.S, self.K, self.T, self.r, self.q,
                                     self.sigma, 1, n_steps=500))
        fd = float(fd_black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                           self.sigma, 1, n_space=400, n_time=400))

        assert abs(bsm - tree) < 0.05, f"BSM={bsm}, Tree={tree}"
        assert abs(bsm - fd) < 0.10, f"BSM={bsm}, FD={fd}"

    def test_put_call_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T)."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        call = float(black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                          self.sigma, 1))
        put = float(black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                         self.sigma, -1))
        parity = self.S * jnp.exp(-self.q * self.T) - self.K * jnp.exp(-self.r * self.T)
        assert abs(call - put - float(parity)) < 1e-10

    def test_ad_greeks_vs_finite_diff(self):
        """AD delta matches finite-difference delta."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        price_fn = lambda s: black_scholes_price(
            s, self.K, self.T, self.r, self.q, self.sigma, 1
        )
        ad_delta = float(jax.grad(price_fn)(jnp.float64(self.S)))

        eps = 0.01
        fd_delta = float(
            (price_fn(self.S + eps) - price_fn(self.S - eps)) / (2 * eps)
        )
        assert abs(ad_delta - fd_delta) < 1e-6

    def test_vmap_batch_pricing(self):
        """vmap over strikes matches sequential pricing."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])

        batch = jax.vmap(
            lambda k: black_scholes_price(self.S, k, self.T, self.r, self.q,
                                           self.sigma, 1)
        )(strikes)

        sequential = jnp.array([
            black_scholes_price(self.S, float(k), self.T, self.r, self.q,
                                 self.sigma, 1)
            for k in strikes
        ])

        assert jnp.allclose(batch, sequential, atol=1e-12)

    def test_heston_near_bsm(self):
        """Heston with near-zero vol-of-vol → BSM."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        from ql_jax.engines.analytic.heston import heston_price

        bsm = float(black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                          self.sigma, 1))
        v0 = self.sigma**2
        heston = float(heston_price(self.S, self.K, self.T, self.r, self.q,
                                     v0, 1.0, v0, 0.01, 0.0, 1))
        assert abs(bsm - heston) < 0.20

    def test_american_geq_european(self):
        """American option price >= European option price."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        from ql_jax.engines.lattice.binomial import binomial_price

        euro_put = float(black_scholes_price(self.S, self.K, self.T, self.r, self.q,
                                              self.sigma, -1))
        amer_put = float(binomial_price(self.S, self.K, self.T, self.r, self.q,
                                         self.sigma, -1, n_steps=300, american=True))
        assert amer_put >= euro_put - 0.001


class TestMCConvergence:
    """Monte Carlo convergence to analytic."""

    def test_mc_european_converges(self):
        """MC European price converges to BSM with enough paths."""
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        from ql_jax.engines.mc.european import mc_european_bs

        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
        bsm = float(black_scholes_price(S, K, T, r, q, sigma, 1))
        price, stderr = mc_european_bs(S, K, T, r, q, sigma, 1, n_paths=100_000,
                                        key=jax.random.PRNGKey(42))
        assert abs(float(price) - bsm) < 0.50  # within 50 cents on 100$ option


class TestJITCorrectness:
    """JIT output matches eager output."""

    def test_jit_bsm(self):
        from ql_jax.engines.analytic.black_formula import black_scholes_price

        args = (100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 1)
        eager = float(black_scholes_price(*args))
        jitted = float(jax.jit(black_scholes_price, static_argnums=(6,))(*args))
        assert eager == jitted

    def test_jit_binomial(self):
        from ql_jax.engines.lattice.binomial import binomial_price

        eager = float(binomial_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 1, n_steps=50))
        jitted = float(jax.jit(binomial_price, static_argnums=(6, 7))
                       (100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 1, 50))
        assert abs(eager - jitted) < 1e-12
