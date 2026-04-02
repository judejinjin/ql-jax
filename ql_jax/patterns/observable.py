"""Observable pattern — functional replacement for QuantLib's Observer/Observable.

In QuantLib, Observer/Observable provides mutable-state notification for market data
changes. In JAX's functional paradigm, we replace this with explicit dependency passing:
when market data changes, create new immutable state and re-call pricing functions.

This module provides lightweight helpers for the pattern.
"""


class Settings:
    """Global evaluation settings (singleton-like).

    Replaces QuantLib's Settings::instance(). In the functional JAX world,
    these are explicit parameters passed to pricing functions rather than
    global mutable state. This class is a bridge for convenience.
    """

    _evaluation_date = None

    @classmethod
    def evaluation_date(cls):
        return cls._evaluation_date

    @classmethod
    def set_evaluation_date(cls, d):
        cls._evaluation_date = d


def recalculate(pricing_fn, market_data):
    """Re-evaluate a pricing function with updated market data.

    This is the functional equivalent of Observable::notifyObservers():
    instead of mutating state and notifying watchers, we simply re-call
    the pure function with new inputs.
    """
    return pricing_fn(market_data)
