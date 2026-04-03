"""Swap spread index: difference of two swap rates.

Used for pricing CMS spread coupons and range accruals.
"""

from __future__ import annotations

from dataclasses import dataclass
from ql_jax.indexes.swap import SwapIndex


@dataclass
class SwapSpreadIndex:
    """Index representing the spread between two swap rates.

    spread(t) = swap_index_1(t) - swap_index_2(t)
    """
    index1: SwapIndex
    index2: SwapIndex
    family_name: str = "SwapSpread"

    @property
    def name(self) -> str:
        return f"{self.index1.name}-{self.index2.name}"

    def fixing(self, date, forecast_curve1=None, forecast_curve2=None):
        """Return swap spread fixing: rate1 - rate2."""
        r1 = self.index1.fixing(date, forecast_curve1)
        r2 = self.index2.fixing(date, forecast_curve2)
        return r1 - r2
