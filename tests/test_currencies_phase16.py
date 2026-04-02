"""Tests for Phase 16 currencies and exchange rates, and Phase 17 settings."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Currencies
# ---------------------------------------------------------------------------

class TestAmericaCurrencies:
    def test_usd(self):
        from ql_jax.currencies.america import USD
        assert USD.code == "USD"
        assert USD.numeric_code == 840
        assert USD.symbol == "$"
        assert USD.fractions_per_unit == 100

    def test_brl(self):
        from ql_jax.currencies.america import BRL
        assert BRL.code == "BRL"

    def test_cad(self):
        from ql_jax.currencies.america import CAD
        assert CAD.code == "CAD"


class TestEuropeCurrencies:
    def test_eur(self):
        from ql_jax.currencies.europe import EUR
        assert EUR.code == "EUR"
        assert EUR.numeric_code == 978

    def test_gbp(self):
        from ql_jax.currencies.europe import GBP
        assert GBP.code == "GBP"

    def test_chf(self):
        from ql_jax.currencies.europe import CHF
        assert CHF.code == "CHF"


class TestAsiaCurrencies:
    def test_jpy(self):
        from ql_jax.currencies.asia import JPY
        assert JPY.code == "JPY"
        assert JPY.fractions_per_unit == 1  # No subunit

    def test_cny(self):
        from ql_jax.currencies.asia import CNY
        assert CNY.code == "CNY"


class TestAfricaCurrencies:
    def test_zar(self):
        from ql_jax.currencies.africa import ZAR
        assert ZAR.code == "ZAR"


class TestOceaniaCurrencies:
    def test_aud(self):
        from ql_jax.currencies.oceania import AUD
        assert AUD.code == "AUD"

    def test_nzd(self):
        from ql_jax.currencies.oceania import NZD
        assert NZD.code == "NZD"


# ---------------------------------------------------------------------------
# Exchange rates
# ---------------------------------------------------------------------------

class TestExchangeRate:
    def test_exchange(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR
        rate = ExchangeRate(source=USD, target=EUR, rate=0.92)
        assert rate.exchange(100.0) == pytest.approx(92.0)

    def test_inverse(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR
        rate = ExchangeRate(source=USD, target=EUR, rate=0.92)
        inv = rate.inverse()
        assert inv.source == EUR
        assert inv.target == USD
        assert inv.rate == pytest.approx(1.0 / 0.92)

    def test_repr(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR
        rate = ExchangeRate(source=USD, target=EUR, rate=0.92)
        assert "USD" in repr(rate)
        assert "EUR" in repr(rate)


class TestExchangeRateManager:
    def test_direct_lookup(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate, ExchangeRateManager
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR
        mgr = ExchangeRateManager()
        mgr.clear()
        mgr.add(ExchangeRate(source=USD, target=EUR, rate=0.92))
        result = mgr.lookup("USD", "EUR")
        assert result is not None
        assert result.rate == pytest.approx(0.92)

    def test_inverse_lookup(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate, ExchangeRateManager
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR
        mgr = ExchangeRateManager()
        mgr.clear()
        mgr.add(ExchangeRate(source=USD, target=EUR, rate=0.92))
        result = mgr.lookup("EUR", "USD")
        assert result is not None
        assert result.rate == pytest.approx(1.0 / 0.92, rel=0.001)

    def test_triangulation(self):
        from ql_jax.currencies.exchange_rate import ExchangeRate, ExchangeRateManager
        from ql_jax.currencies.america import USD
        from ql_jax.currencies.europe import EUR, GBP
        mgr = ExchangeRateManager()
        mgr.clear()
        mgr.add(ExchangeRate(source=USD, target=EUR, rate=0.92))
        mgr.add(ExchangeRate(source=USD, target=GBP, rate=0.79))
        result = mgr.lookup("EUR", "GBP")
        if result is not None:
            # EUR->USD->GBP: (1/0.92) * 0.79 ~ 0.8587
            assert result.rate == pytest.approx(0.79 / 0.92, rel=0.01)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_singleton(self):
        from ql_jax.settings import Settings
        s1 = Settings.instance()
        s2 = Settings.instance()
        assert s1 is s2

    def test_evaluation_date(self):
        from ql_jax.settings import Settings
        from ql_jax.time.date import Date
        s = Settings.instance()
        d = Date(15, 6, 2024)
        s.evaluation_date = d
        assert s.evaluation_date == d

    def test_saved_settings(self):
        from ql_jax.settings import Settings, SavedSettings
        from ql_jax.time.date import Date
        s = Settings.instance()
        original = s.evaluation_date
        with SavedSettings():
            s.evaluation_date = Date(1, 1, 2020)
            assert s.evaluation_date == Date(1, 1, 2020)
        assert s.evaluation_date == original
