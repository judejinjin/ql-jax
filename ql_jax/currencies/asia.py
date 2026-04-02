"""Asian currencies."""

from __future__ import annotations

from ql_jax.currencies.america import Currency

JPY = Currency("Japanese Yen", "JPY", 392, "¥", "", 1)
CNY = Currency("Chinese Yuan", "CNY", 156, "¥", "fen", 100)
HKD = Currency("Hong Kong Dollar", "HKD", 344, "HK$", "cent", 100)
SGD = Currency("Singapore Dollar", "SGD", 702, "S$", "cent", 100)
KRW = Currency("Korean Won", "KRW", 410, "₩", "jeon", 100)
INR = Currency("Indian Rupee", "INR", 356, "₹", "paisa", 100)
TWD = Currency("Taiwan Dollar", "TWD", 901, "NT$", "cent", 100)
THB = Currency("Thai Baht", "THB", 764, "฿", "satang", 100)
IDR = Currency("Indonesian Rupiah", "IDR", 360, "Rp", "sen", 100)
MYR = Currency("Malaysian Ringgit", "MYR", 458, "RM", "sen", 100)
PHP = Currency("Philippine Peso", "PHP", 608, "₱", "centavo", 100)
VND = Currency("Vietnamese Dong", "VND", 704, "₫", "hào", 10)
PKR = Currency("Pakistani Rupee", "PKR", 586, "Rs", "paisa", 100)
BDT = Currency("Bangladeshi Taka", "BDT", 50, "৳", "poisha", 100)
ILS = Currency("Israeli Shekel", "ILS", 376, "₪", "agora", 100)
SAR = Currency("Saudi Riyal", "SAR", 682, "SAR", "halala", 100)
AED = Currency("UAE Dirham", "AED", 784, "AED", "fils", 100)
QAR = Currency("Qatari Riyal", "QAR", 634, "QAR", "dirham", 100)
KWD = Currency("Kuwaiti Dinar", "KWD", 414, "KWD", "fils", 1000)
