"""Americas currencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    """Currency definition."""
    name: str
    code: str
    numeric_code: int
    symbol: str
    fractional_unit: str
    fractions_per_unit: int
    rounding_type: str = "closest"


# Americas
USD = Currency("US Dollar", "USD", 840, "$", "cent", 100)
CAD = Currency("Canadian Dollar", "CAD", 124, "C$", "cent", 100)
BRL = Currency("Brazilian Real", "BRL", 986, "R$", "centavo", 100)
MXN = Currency("Mexican Peso", "MXN", 484, "Mex$", "centavo", 100)
ARS = Currency("Argentine Peso", "ARS", 32, "$", "centavo", 100)
CLP = Currency("Chilean Peso", "CLP", 152, "$", "centavo", 100)
COP = Currency("Colombian Peso", "COP", 170, "$", "centavo", 100)
PEN = Currency("Peruvian Sol", "PEN", 604, "S/", "céntimo", 100)
UYU = Currency("Uruguayan Peso", "UYU", 858, "$U", "centésimo", 100)
VEB = Currency("Venezuelan Bolívar", "VEB", 862, "Bs.F", "céntimo", 100)
TTD = Currency("Trinidad & Tobago Dollar", "TTD", 780, "TT$", "cent", 100)
