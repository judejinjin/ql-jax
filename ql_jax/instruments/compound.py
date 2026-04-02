"""Compound option — option on an option."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class CompoundOption:
    """Compound option: an option whose underlying is another option.

    Parameters
    ----------
    strike_mother : strike of the mother (outer) option
    strike_daughter : strike of the daughter (underlying) option
    maturity_mother : expiry of the mother option
    maturity_daughter : expiry of the daughter option
    is_call_mother : True if mother is a call
    is_call_daughter : True if daughter is a call
    """
    strike_mother: float
    strike_daughter: float
    maturity_mother: float
    maturity_daughter: float
    is_call_mother: bool = True
    is_call_daughter: bool = True
