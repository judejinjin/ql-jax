"""Tests for lattice framework."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestLattice1D:
    def test_binomial_construction(self):
        from ql_jax.methods.lattices.lattice import Lattice1D
        lat = Lattice1D(n_steps=100, dt=0.01, tree_type='binomial')
        assert lat.n_steps == 100
        assert lat.dt == pytest.approx(0.01)

    def test_trinomial_construction(self):
        from ql_jax.methods.lattices.lattice import Lattice1D
        lat = Lattice1D(n_steps=50, dt=0.02, tree_type='trinomial')
        assert lat.n_steps == 50

    def test_rollback_preserves_shape(self):
        from ql_jax.methods.lattices.lattice import Lattice1D
        lat = Lattice1D(n_steps=10, dt=0.1, tree_type='binomial')
        # For binomial tree at step i, there are i+1 nodes
        values = jnp.ones(11)  # step 10 has 11 nodes
        discount = 0.99
        rolled = lat.rollback(values, step=10, discount=discount)
        assert rolled.shape[0] == 10  # step 9 has 10 nodes

    def test_full_rollback(self):
        from ql_jax.methods.lattices.lattice import Lattice1D
        lat = Lattice1D(n_steps=100, dt=0.01, tree_type='binomial')
        # Constant terminal values
        terminal = jnp.ones(101)
        discount_fn = lambda step: 0.999
        price = lat.full_rollback(terminal, discount_fn)
        # With discount ~0.999 over 100 steps ~= 0.999^100 ~ 0.905
        assert float(price) == pytest.approx(0.999 ** 100, rel=0.05)


class TestLattice2D:
    def test_construction(self):
        from ql_jax.methods.lattices.lattice import Lattice2D
        lat = Lattice2D(n_steps=50, dt=0.02, correlation=0.3)
        assert lat.correlation == pytest.approx(0.3)

    def test_rollback_2d(self):
        from ql_jax.methods.lattices.lattice import Lattice2D
        lat = Lattice2D(n_steps=5, dt=0.2, correlation=0.0)
        # At step 5, nodes: (6, 6) for uncorrelated 2D binomial
        terminal = jnp.ones((6, 6))
        rolled = lat.rollback(terminal, step=5, discount=0.99)
        assert rolled.shape == (5, 5)


class TestTwoFactorLattice:
    def test_construction(self):
        from ql_jax.methods.lattices.lattice import TwoFactorLattice
        lat = TwoFactorLattice(n_steps=50, dt=0.02, sigma1=0.01, sigma2=0.008,
                                kappa1=0.1, kappa2=0.2, rho=-0.5)
        grids = lat.build_grid()
        assert len(grids) == 2
        assert grids[0].shape[0] > 0
        assert grids[1].shape[0] > 0

    def test_pricing(self):
        from ql_jax.methods.lattices.lattice import TwoFactorLattice
        lat = TwoFactorLattice(n_steps=20, dt=0.05, sigma1=0.01, sigma2=0.008,
                                kappa1=0.1, kappa2=0.2, rho=-0.3)
        terminal_fn = lambda x1, x2: jnp.maximum(1.0 - jnp.exp(-(x1 + x2) * 5), 0.0)
        discount_fn = lambda step: jnp.exp(-0.03 * 0.05)
        price = lat.price(terminal_fn, discount_fn)
        assert float(price) >= 0
