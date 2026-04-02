"""African currencies."""

from __future__ import annotations

from ql_jax.currencies.america import Currency

ZAR = Currency("South African Rand", "ZAR", 710, "R", "cent", 100)
NGN = Currency("Nigerian Naira", "NGN", 566, "₦", "kobo", 100)
KES = Currency("Kenyan Shilling", "KES", 404, "KSh", "cent", 100)
EGP = Currency("Egyptian Pound", "EGP", 818, "E£", "piastre", 100)
GHS = Currency("Ghanaian Cedi", "GHS", 936, "GH₵", "pesewa", 100)
MAD = Currency("Moroccan Dirham", "MAD", 504, "MAD", "centime", 100)
TND = Currency("Tunisian Dinar", "TND", 788, "TND", "millime", 1000)
BWP = Currency("Botswana Pula", "BWP", 72, "P", "thebe", 100)
MUR = Currency("Mauritian Rupee", "MUR", 480, "Rs", "cent", 100)
