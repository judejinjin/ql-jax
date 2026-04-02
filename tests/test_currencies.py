"""Tests for currencies."""

from ql_jax.currencies import (
    Currency, ExchangeRate, ExchangeRateManager,
    USD, EUR, GBP, JPY, CHF,
)


class TestCurrency:
    def test_usd(self):
        assert USD.code == "USD"
        assert USD.numeric_code == 840
        assert USD.symbol == "$"
        assert USD.fractions_per_unit == 100

    def test_eur(self):
        assert EUR.code == "EUR"
        assert EUR.numeric_code == 978

    def test_jpy(self):
        assert JPY.fractions_per_unit == 1

    def test_equality(self):
        usd2 = Currency("US Dollar", "USD", 840, "$", 100)
        assert USD == usd2

    def test_hash(self):
        d = {USD: 1, EUR: 2}
        assert d[USD] == 1

    def test_repr(self):
        assert repr(USD) == "USD"


class TestExchangeRate:
    def test_exchange(self):
        rate = ExchangeRate(EUR, USD, 1.10)
        assert abs(rate.exchange(100.0) - 110.0) < 1e-10

    def test_inverse(self):
        rate = ExchangeRate(EUR, USD, 1.10)
        inv = rate.inverse()
        assert inv.source == USD
        assert inv.target == EUR
        assert abs(inv.rate - 1.0 / 1.10) < 1e-10


class TestExchangeRateManager:
    def test_lookup(self):
        mgr = ExchangeRateManager()
        mgr.add(ExchangeRate(EUR, USD, 1.10))
        rate = mgr.lookup(EUR, USD)
        assert abs(rate.rate - 1.10) < 1e-10

    def test_inverse_lookup(self):
        mgr = ExchangeRateManager()
        mgr.add(ExchangeRate(EUR, USD, 1.10))
        rate = mgr.lookup(USD, EUR)
        assert abs(rate.rate - 1.0 / 1.10) < 1e-10
