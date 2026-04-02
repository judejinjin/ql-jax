"""Inflation term structures."""

from ql_jax.termstructures.inflation.curves import (
    InterpolatedZeroInflationCurve,
    InterpolatedYoYInflationCurve,
    Seasonality,
)
from ql_jax.termstructures.inflation.piecewise import (
    PiecewiseZeroInflationCurve,
    PiecewiseYoYInflationCurve,
    bootstrap_zero_inflation,
    bootstrap_yoy_inflation,
)
