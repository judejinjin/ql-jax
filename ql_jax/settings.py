"""Global settings: evaluation date, etc."""

from __future__ import annotations

from ql_jax.time.date import Date


class Settings:
    """Global settings singleton.

    Primary use: evaluation_date — the "today" date for all pricing.

    Usage:
        Settings.instance().evaluation_date = Date(15, 6, 2024)
        today = Settings.instance().evaluation_date
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._evaluation_date = None
        return cls._instance

    @classmethod
    def instance(cls) -> 'Settings':
        """Return the singleton instance."""
        return cls()

    @property
    def evaluation_date(self) -> Date:
        """Get the current evaluation date."""
        if self._evaluation_date is None:
            # Default: today's date
            import datetime
            d = datetime.date.today()
            return Date(d.day, d.month, d.year)
        return self._evaluation_date

    @evaluation_date.setter
    def evaluation_date(self, d: Date):
        """Set the evaluation date."""
        self._evaluation_date = d


class SavedSettings:
    """Context manager to save and restore Settings.

    Usage:
        with SavedSettings():
            Settings.instance().evaluation_date = Date(1, 1, 2020)
            # ... pricing ...
        # Settings restored to previous state
    """

    def __enter__(self):
        self._saved_date = Settings.instance()._evaluation_date
        return self

    def __exit__(self, *args):
        Settings.instance()._evaluation_date = self._saved_date
