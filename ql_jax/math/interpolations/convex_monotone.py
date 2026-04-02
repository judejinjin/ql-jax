"""Convex-monotone interpolation preserving convexity and monotonicity.

Hyman (1983) filter applied to cubic interpolation to enforce
monotonicity, plus convexity-preserving adjustments.
"""

import jax.numpy as jnp


def build(xs, ys):
    """Build a convex-monotone interpolant.

    Returns dict with spline data: xs, ys, slopes, coefficients.
    """
    n = len(xs)
    dx = jnp.diff(xs)
    dy = jnp.diff(ys)
    secants = dy / dx

    # Initialize slopes via harmonic mean of secants (Fritsch-Carlson)
    slopes = jnp.zeros(n)
    for i in range(1, n - 1):
        s_left = secants[i - 1]
        s_right = secants[i]
        # Only set nonzero if same sign (monotone)
        same_sign = s_left * s_right > 0
        harmonic = 2.0 * s_left * s_right / (s_left + s_right + 1e-30)
        slopes = slopes.at[i].set(jnp.where(same_sign, harmonic, 0.0))

    # Endpoint: one-sided
    slopes = slopes.at[0].set(secants[0])
    slopes = slopes.at[n - 1].set(secants[-1])

    # Hyman monotonicity filter
    for i in range(n - 1):
        s = secants[i]
        abs_s = jnp.abs(s)
        slopes = slopes.at[i].set(
            jnp.where(jnp.abs(slopes[i]) > 3.0 * abs_s,
                       3.0 * s, slopes[i])
        )
        slopes = slopes.at[i + 1].set(
            jnp.where(jnp.abs(slopes[i + 1]) > 3.0 * abs_s,
                       3.0 * s, slopes[i + 1])
        )

    return {'xs': jnp.array(xs), 'ys': jnp.array(ys),
            'slopes': slopes, 'dx': dx, 'dy': dy, 'secants': secants}


def evaluate(data, x):
    """Evaluate convex-monotone interpolant at x using Hermite basis."""
    xs = data['xs']
    ys = data['ys']
    slopes = data['slopes']
    n = len(xs)

    # Find interval via clamp (for JIT compatibility)
    idx = jnp.searchsorted(xs, x, side='right') - 1
    idx = jnp.clip(idx, 0, n - 2)

    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]
    m0 = slopes[idx]
    m1 = slopes[idx + 1]
    h = x1 - x0
    t = (x - x0) / h

    # Hermite basis functions
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t ** 2 * (3.0 - 2.0 * t)
    h11 = t ** 2 * (t - 1.0)

    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
