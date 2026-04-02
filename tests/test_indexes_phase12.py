"""Tests for Phase 12 indexes: manager and new IBOR indexes."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Index Manager
# ---------------------------------------------------------------------------

class TestIndexManager:
    def test_set_and_get_fixing(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.clear_fixings()
        d = Date(15, 1, 2024)
        mgr.set_fixing("SOFR", d, 0.051)
        assert mgr.has_fixing("SOFR", d)
        assert mgr.get_fixing("SOFR", d) == pytest.approx(0.051)

    def test_missing_fixing(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.clear_fixings()
        assert not mgr.has_fixing("UNKNOWN", Date(1, 1, 2024))
        assert mgr.get_fixing("UNKNOWN", Date(1, 1, 2024)) is None

    def test_set_fixings_bulk(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.clear_fixings()
        dates = [Date(15, 1, 2024), Date(16, 1, 2024), Date(17, 1, 2024)]
        values = [0.05, 0.051, 0.052]
        mgr.set_fixings("Euribor6M", dates, values)
        assert mgr.has_fixing("Euribor6M", dates[0])
        assert mgr.get_fixing("Euribor6M", dates[2]) == pytest.approx(0.052)

    def test_clear_fixings(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.set_fixing("TEST", Date(1, 6, 2024), 0.01)
        mgr.clear_fixings("TEST")
        assert not mgr.has_fixing("TEST", Date(1, 6, 2024))

    def test_clear_all(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.set_fixing("A", Date(1, 1, 2024), 0.01)
        mgr.set_fixing("B", Date(1, 1, 2024), 0.02)
        mgr.clear_fixings()
        assert not mgr.has_fixing("A", Date(1, 1, 2024))
        assert not mgr.has_fixing("B", Date(1, 1, 2024))

    def test_all_fixings(self):
        from ql_jax.indexes.manager import IndexManager
        from ql_jax.time.date import Date
        mgr = IndexManager()
        mgr.clear_fixings()
        mgr.set_fixing("SONIA", Date(1, 3, 2024), 0.05)
        mgr.set_fixing("SONIA", Date(4, 3, 2024), 0.051)
        serials, values = mgr.all_fixings("SONIA")
        assert serials is not None
        assert len(values) == 2


# ---------------------------------------------------------------------------
# New IBOR/Overnight indexes
# ---------------------------------------------------------------------------

class TestNewIborIndexes:
    """Smoke tests for newly added index factories."""

    def test_aud_libor(self):
        from ql_jax.indexes.ibor import AUDLibor
        idx = AUDLibor(3)
        assert idx.name is not None

    def test_bbsw(self):
        from ql_jax.indexes.ibor import BBSW
        idx = BBSW(3)
        assert idx.name is not None

    def test_cad_libor(self):
        from ql_jax.indexes.ibor import CADLibor
        idx = CADLibor(3)
        assert idx.name is not None

    def test_cdor(self):
        from ql_jax.indexes.ibor import CDOR
        idx = CDOR(3)
        assert idx.name is not None

    def test_cdi(self):
        from ql_jax.indexes.ibor import CDI
        idx = CDI()
        assert idx.name is not None

    def test_destr(self):
        from ql_jax.indexes.ibor import DESTR
        idx = DESTR()
        assert idx.name is not None

    def test_dkk_libor(self):
        from ql_jax.indexes.ibor import DKKLibor
        idx = DKKLibor(3)
        assert idx.name is not None

    def test_jibar(self):
        from ql_jax.indexes.ibor import Jibar
        idx = Jibar(3)
        assert idx.name is not None

    def test_kofr(self):
        from ql_jax.indexes.ibor import KOFR
        idx = KOFR()
        assert idx.name is not None

    def test_nzd_libor(self):
        from ql_jax.indexes.ibor import NZDLibor
        idx = NZDLibor(3)
        assert idx.name is not None

    def test_nzocr(self):
        from ql_jax.indexes.ibor import NZOCR
        idx = NZOCR()
        assert idx.name is not None

    def test_pribor(self):
        from ql_jax.indexes.ibor import Pribor
        idx = Pribor(3)
        assert idx.name is not None

    def test_sek_libor(self):
        from ql_jax.indexes.ibor import SEKLibor
        idx = SEKLibor(3)
        assert idx.name is not None

    def test_shibor(self):
        from ql_jax.indexes.ibor import Shibor
        idx = Shibor(3)
        assert idx.name is not None

    def test_swestr(self):
        from ql_jax.indexes.ibor import SWESTR
        idx = SWESTR()
        assert idx.name is not None

    def test_tonar(self):
        from ql_jax.indexes.ibor import Tonar
        idx = Tonar()
        assert idx.name is not None

    def test_wibor(self):
        from ql_jax.indexes.ibor import Wibor
        idx = Wibor(3)
        assert idx.name is not None
