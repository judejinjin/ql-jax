"""Currency definitions.

QuantLib Currency is metadata — name, code, numeric code, symbol, rounding.
Non-differentiable. Frozen dataclass for immutability.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    """Currency specification."""
    name: str
    code: str
    numeric_code: int
    symbol: str
    fractions_per_unit: int = 100

    def __repr__(self):
        return self.code

    def __eq__(self, other):
        if isinstance(other, Currency):
            return self.code == other.code
        return NotImplemented

    def __hash__(self):
        return hash(self.code)


@dataclass(frozen=True)
class ExchangeRate:
    """Exchange rate between two currencies."""
    source: Currency
    target: Currency
    rate: float

    def exchange(self, amount: float) -> float:
        return amount * self.rate

    def inverse(self) -> ExchangeRate:
        return ExchangeRate(self.target, self.source, 1.0 / self.rate)


class ExchangeRateManager:
    """Registry of exchange rates."""

    def __init__(self):
        self._rates: dict[tuple[str, str], ExchangeRate] = {}

    def add(self, rate: ExchangeRate):
        self._rates[(rate.source.code, rate.target.code)] = rate

    def lookup(self, source: Currency, target: Currency) -> ExchangeRate:
        key = (source.code, target.code)
        if key in self._rates:
            return self._rates[key]
        inv_key = (target.code, source.code)
        if inv_key in self._rates:
            return self._rates[inv_key].inverse()
        raise KeyError(f"No exchange rate for {source.code}/{target.code}")


# ---------------------------------------------------------------------------
# Major currencies
# ---------------------------------------------------------------------------

USD = Currency("US Dollar", "USD", 840, "$", 100)
EUR = Currency("Euro", "EUR", 978, "€", 100)
GBP = Currency("British Pound", "GBP", 826, "£", 100)
JPY = Currency("Japanese Yen", "JPY", 392, "¥", 1)
CHF = Currency("Swiss Franc", "CHF", 756, "CHF", 100)
CAD = Currency("Canadian Dollar", "CAD", 124, "C$", 100)
AUD = Currency("Australian Dollar", "AUD", 36, "A$", 100)
NZD = Currency("New Zealand Dollar", "NZD", 554, "NZ$", 100)
SEK = Currency("Swedish Krona", "SEK", 752, "kr", 100)
NOK = Currency("Norwegian Krone", "NOK", 578, "kr", 100)
DKK = Currency("Danish Krone", "DKK", 208, "kr", 100)
HKD = Currency("Hong Kong Dollar", "HKD", 344, "HK$", 100)
SGD = Currency("Singapore Dollar", "SGD", 702, "S$", 100)
CNY = Currency("Chinese Yuan", "CNY", 156, "¥", 100)
INR = Currency("Indian Rupee", "INR", 356, "₹", 100)
KRW = Currency("South Korean Won", "KRW", 410, "₩", 1)
BRL = Currency("Brazilian Real", "BRL", 986, "R$", 100)
MXN = Currency("Mexican Peso", "MXN", 484, "$", 100)
ZAR = Currency("South African Rand", "ZAR", 710, "R", 100)
TRY = Currency("Turkish Lira", "TRY", 949, "₺", 100)
PLN = Currency("Polish Zloty", "PLN", 985, "zł", 100)
CZK = Currency("Czech Koruna", "CZK", 203, "Kč", 100)
HUF = Currency("Hungarian Forint", "HUF", 348, "Ft", 1)
RON = Currency("Romanian Leu", "RON", 946, "lei", 100)
THB = Currency("Thai Baht", "THB", 764, "฿", 100)
TWD = Currency("Taiwan Dollar", "TWD", 901, "NT$", 100)
