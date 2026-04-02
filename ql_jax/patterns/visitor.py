"""Visitor pattern — using functools.singledispatch.

QuantLib's Visitor pattern (for instrument traversal, etc.) maps naturally
to Python's singledispatch for type-based dispatch.
"""

import functools


def make_visitor():
    """Create a visitor using singledispatch.

    Usage:
        visit = make_visitor()

        @visit.register(Bond)
        def _(instrument):
            ...

        @visit.register(Swap)
        def _(instrument):
            ...

        visit(some_instrument)
    """
    @functools.singledispatch
    def visit(obj):
        raise NotImplementedError(f"No visitor registered for {type(obj).__name__}")

    return visit
