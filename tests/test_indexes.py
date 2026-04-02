"""Tests for indexes."""

from ql_jax.indexes.ibor import (
    IborIndex, OvernightIndex, USDLibor, SOFR, FedFunds,
    Euribor, ESTR, GBPLibor, SONIA, JPYLibor, TONA,
)
from ql_jax.indexes.swap import SwapIndex, EuriborSwapIsdaFixA
from ql_jax.time.date import Date


class TestIborIndex:
    def test_usd_libor(self):
        idx = USDLibor(3)
        assert idx.family_name == "USDLibor"
        assert idx.tenor_months == 3
        assert idx.name == "USDLibor3M"
        assert idx.currency_code == "USD"

    def test_euribor(self):
        idx = Euribor(6)
        assert idx.name == "Euribor6M"
        assert idx.fixing_days == 2

    def test_gbp_libor(self):
        idx = GBPLibor(3)
        assert idx.fixing_days == 0
        assert idx.day_counter_convention == "Actual365Fixed"

    def test_add_fixing(self):
        idx = USDLibor(3)
        d = Date(15, 6, 2023)
        idx.add_fixing(d, 0.05)
        assert abs(idx.fixing(d) - 0.05) < 1e-12

    def test_add_fixings(self):
        idx = Euribor(6)
        d1 = Date(14, 6, 2023)
        d2 = Date(15, 6, 2023)
        idx.add_fixings([d1, d2], [0.03, 0.031])
        assert abs(idx.fixing(d1) - 0.03) < 1e-12
        assert abs(idx.fixing(d2) - 0.031) < 1e-12

    def test_clear_fixings(self):
        idx = USDLibor(3)
        d = Date(15, 6, 2023)
        idx.add_fixing(d, 0.05)
        idx.clear_fixings()
        try:
            idx.fixing(d)
            assert False, "Should raise KeyError"
        except KeyError:
            pass


class TestOvernightIndex:
    def test_sofr(self):
        idx = SOFR()
        assert idx.family_name == "SOFR"
        assert idx.tenor_months == 0
        assert idx.name == "SOFR"

    def test_sonia(self):
        idx = SONIA()
        assert idx.currency_code == "GBP"

    def test_estr(self):
        idx = ESTR()
        assert idx.currency_code == "EUR"

    def test_tona(self):
        idx = TONA()
        assert idx.currency_code == "JPY"


class TestSwapIndex:
    def test_euribor_swap(self):
        idx = EuriborSwapIsdaFixA(10)
        assert idx.tenor_months == 120
        assert idx.fixed_leg_tenor_months == 12
