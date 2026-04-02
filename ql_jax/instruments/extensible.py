"""Extensible options — holder and writer extensible."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class HolderExtensibleOption:
    """Holder-extensible option.

    At the initial expiry, the holder can choose to exercise or
    to extend the option to a later maturity (possibly with a different strike),
    paying an extension premium.

    Parameters
    ----------
    strike : strike of the initial option
    maturity : initial maturity
    extended_strike : strike of the extended option
    extended_maturity : extended maturity
    extension_premium : premium paid to extend
    is_call : True for call
    """
    strike: float
    maturity: float
    extended_strike: float
    extended_maturity: float
    extension_premium: float
    is_call: bool = True


@dataclass(frozen=True)
class WriterExtensibleOption:
    """Writer-extensible option.

    If the option is out-of-the-money at expiry, the writer can
    extend it to a later maturity with a different strike.

    Parameters
    ----------
    strike : initial strike
    maturity : initial maturity
    extended_strike : strike after extension
    extended_maturity : maturity after extension
    is_call : True for call
    """
    strike: float
    maturity: float
    extended_strike: float
    extended_maturity: float
    is_call: bool = True
