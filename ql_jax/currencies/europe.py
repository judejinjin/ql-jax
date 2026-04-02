"""European currencies."""

from __future__ import annotations

from ql_jax.currencies.america import Currency

EUR = Currency("Euro", "EUR", 978, "€", "cent", 100)
GBP = Currency("British Pound", "GBP", 826, "£", "penny", 100)
CHF = Currency("Swiss Franc", "CHF", 756, "CHF", "rappen", 100)
SEK = Currency("Swedish Krona", "SEK", 752, "kr", "öre", 100)
NOK = Currency("Norwegian Krone", "NOK", 578, "kr", "øre", 100)
DKK = Currency("Danish Krone", "DKK", 208, "kr", "øre", 100)
PLN = Currency("Polish Zloty", "PLN", 985, "zł", "grosz", 100)
CZK = Currency("Czech Koruna", "CZK", 203, "Kč", "haléř", 100)
HUF = Currency("Hungarian Forint", "HUF", 348, "Ft", "fillér", 100)
RON = Currency("Romanian Leu", "RON", 946, "lei", "ban", 100)
RUB = Currency("Russian Ruble", "RUB", 643, "₽", "kopeck", 100)
TRY = Currency("Turkish Lira", "TRY", 949, "₺", "kuruş", 100)
ISK = Currency("Icelandic Króna", "ISK", 352, "kr", "aurar", 100)
UAH = Currency("Ukrainian Hryvnia", "UAH", 980, "₴", "kopiyka", 100)
BGN = Currency("Bulgarian Lev", "BGN", 975, "лв", "stotinka", 100)
HRK = Currency("Croatian Kuna", "HRK", 191, "kn", "lipa", 100)
