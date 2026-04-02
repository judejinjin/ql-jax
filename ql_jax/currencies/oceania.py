"""Oceania currencies."""

from __future__ import annotations

from ql_jax.currencies.america import Currency

AUD = Currency("Australian Dollar", "AUD", 36, "A$", "cent", 100)
NZD = Currency("New Zealand Dollar", "NZD", 554, "NZ$", "cent", 100)
FJD = Currency("Fiji Dollar", "FJD", 242, "FJ$", "cent", 100)
PGK = Currency("Papua New Guinean Kina", "PGK", 598, "K", "toea", 100)
