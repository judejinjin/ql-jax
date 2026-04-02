"""Tests for FD framework, cashflows, calendars, and indexes."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date


# ── FD Framework ─────────────────────────────────────────────────────────────

class TestMeshers:
    def test_uniform_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import Uniform1dMesher
        m = Uniform1dMesher(low=0.0, high=1.0, size=11)
        locs = m.locations()
        assert locs.shape == (11,)
        assert abs(float(locs[0]) - 0.0) < 1e-10
        assert abs(float(locs[-1]) - 1.0) < 1e-10

    def test_concentrating_mesher(self):
        from ql_jax.methods.finitedifferences.meshers import Concentrating1dMesher
        m = Concentrating1dMesher(low=0.0, high=5.0, size=51, center=2.5, density=0.1)
        locs = m.locations()
        assert locs.shape == (51,)
        diffs = jnp.diff(locs)
        mid_idx = 25
        assert float(diffs[mid_idx]) < float(diffs[0])


class TestOperators:
    def test_tridiagonal_operator(self):
        from ql_jax.methods.finitedifferences.operators import TridiagonalOperator
        n = 5
        lower = jnp.array([0.0, -1.0, -1.0, -1.0, -1.0])
        diag = jnp.full(n, 2.0)
        upper = jnp.array([-1.0, -1.0, -1.0, -1.0, 0.0])
        op = TridiagonalOperator(lower=lower, diag=diag, upper=upper)
        assert op.lower.shape == (n,)
        assert op.diag.shape == (n,)


class TestBSOperator:
    def test_bs_operator(self):
        from ql_jax.methods.finitedifferences.bs_operator import BSMOperator
        n = 100
        x = jnp.linspace(3.0, 6.0, n)
        op = BSMOperator(r=0.05, q=0.0, sigma=0.20, x_grid=x)
        assert op is not None


# ── Cashflows ────────────────────────────────────────────────────────────────

class TestIndexedCashFlow:
    def test_indexed_cashflow(self):
        from ql_jax.cashflows.indexed import IndexedCashFlow
        cf = IndexedCashFlow(notional=1000.0, base_index=100.0,
                            fixing_date=0.5, payment_date=1.0)
        assert cf.notional == 1000.0


class TestTimeBasket:
    def test_time_basket(self):
        from ql_jax.cashflows.time_basket import TimeBasket
        tb = TimeBasket(bucket_starts=jnp.array([0.0, 0.5, 1.0]),
                       bucket_ends=jnp.array([0.5, 1.0, 1.5]))
        assert tb.bucket_starts.shape == (3,)


# ── New Calendars ────────────────────────────────────────────────────────────

class TestNewCalendars:
    def test_canada_calendar(self):
        from ql_jax.time.calendars import Canada
        cal = Canada()
        d = Date(1, 1, 2024)  # New Year's Day
        assert cal.is_holiday(d)

    def test_australia_calendar(self):
        from ql_jax.time.calendars import Australia
        cal = Australia()
        d = Date(25, 4, 2024)  # ANZAC Day
        assert cal.is_holiday(d)

    def test_china_calendar(self):
        from ql_jax.time.calendars import China
        cal = China()
        d = Date(1, 10, 2024)  # National Day
        assert cal.is_holiday(d)


# ── Inflation Indexes ────────────────────────────────────────────────────────

class TestInflationIndexes:
    def test_uscpi(self):
        from ql_jax.indexes.inflation import USCPI
        idx = USCPI()
        assert hasattr(idx, 'family_name')

    def test_zero_inflation_index(self):
        from ql_jax.indexes.inflation import ZeroInflationIndex
        idx = ZeroInflationIndex(family_name="Test CPI", region="US",
                                revised=False, frequency="Monthly",
                                observation_lag=3)
        assert idx.observation_lag == 3


# ── FX/Equity Indexes ───────────────────────────────────────────────────────

class TestFXEquityIndexes:
    def test_fx_index(self):
        from ql_jax.indexes.fx_equity import EURUSD
        idx = EURUSD()
        assert idx is not None

    def test_equity_index(self):
        from ql_jax.indexes.fx_equity import SPX
        idx = SPX()
        assert idx is not None
