"""IndexManager — global store for index fixings."""

from __future__ import annotations

from typing import Dict, Optional
import jax.numpy as jnp

from ql_jax.time.date import Date


class IndexManager:
    """Singleton-like global store for historical index fixings.

    Usage:
        mgr = IndexManager()
        mgr.set_fixing("Euribor6M", Date(15, 1, 2024), 0.035)
        rate = mgr.get_fixing("Euribor6M", Date(15, 1, 2024))
    """
    _instance = None
    _store: Dict[str, Dict[int, float]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._store = {}
        return cls._instance

    def set_fixing(self, index_name: str, date: Date, value: float):
        """Store a fixing value.

        Parameters
        ----------
        index_name : str
        date : Date
        value : float
        """
        if index_name not in self._store:
            self._store[index_name] = {}
        self._store[index_name][date.serial] = float(value)

    def set_fixings(self, index_name: str, dates, values):
        """Store multiple fixings at once.

        Parameters
        ----------
        index_name : str
        dates : list of Date
        values : list of float
        """
        for d, v in zip(dates, values):
            self.set_fixing(index_name, d, v)

    def get_fixing(self, index_name: str, date: Date) -> Optional[float]:
        """Retrieve a fixing.

        Parameters
        ----------
        index_name : str
        date : Date

        Returns
        -------
        float or None if not found
        """
        if index_name not in self._store:
            return None
        return self._store[index_name].get(date.serial, None)

    def has_fixing(self, index_name: str, date: Date) -> bool:
        """Check if a fixing exists."""
        return self.get_fixing(index_name, date) is not None

    def clear_fixings(self, index_name: str = None):
        """Clear all fixings, or fixings for a specific index.

        Parameters
        ----------
        index_name : str or None (clears all if None)
        """
        if index_name is None:
            self._store.clear()
        elif index_name in self._store:
            del self._store[index_name]

    def all_fixings(self, index_name: str):
        """Return all fixings for an index as (serials, values) arrays.

        Returns
        -------
        (array of serial dates, array of values) or (None, None)
        """
        if index_name not in self._store or not self._store[index_name]:
            return None, None
        items = sorted(self._store[index_name].items())
        serials = jnp.array([s for s, _ in items], dtype=jnp.float64)
        values = jnp.array([v for _, v in items], dtype=jnp.float64)
        return serials, values
