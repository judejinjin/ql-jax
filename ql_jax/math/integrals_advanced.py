"""Advanced numerical integration: Lobatto, Kronrod, tanh-sinh, Filon, 2D, discrete."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gauss_lobatto_integral(f, a, b, n=7):
    """Gauss-Lobatto quadrature (includes endpoints).

    Parameters
    ----------
    f : callable
    a, b : integration bounds
    n : number of points (5 or 7 supported)

    Returns
    -------
    float
    """
    if n == 5:
        nodes = jnp.array([-1.0, -jnp.sqrt(3.0 / 7.0), 0.0, jnp.sqrt(3.0 / 7.0), 1.0])
        weights = jnp.array([1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0])
    else:  # n=7
        nodes = jnp.array([
            -1.0, -0.8302238962785670, -0.4688487934707142, 0.0,
            0.4688487934707142, 0.8302238962785670, 1.0,
        ])
        # Exact: w1=w7=1/21, w2=w6=(124-7√15)/350, w3=w5=(124+7√15)/350, w4=256/525
        sqrt15 = 3.872983346207417
        w_end = 1.0 / 21.0
        w_mid_outer = (124.0 - 7.0 * sqrt15) / 350.0
        w_mid_inner = (124.0 + 7.0 * sqrt15) / 350.0
        w_center = 256.0 / 525.0
        weights = jnp.array([w_end, w_mid_outer, w_mid_inner, w_center,
                             w_mid_inner, w_mid_outer, w_end])

    # Transform from [-1,1] to [a,b]
    half = (b - a) / 2.0
    mid = (a + b) / 2.0
    x = mid + half * nodes
    return half * jnp.sum(weights * jnp.array([f(xi) for xi in x]))


def gauss_kronrod_integral(f, a, b, n=15):
    """Gauss-Kronrod quadrature (7-15 embedded rule).

    Returns both the integral estimate and an error estimate.

    Parameters
    ----------
    f : callable
    a, b : bounds
    n : 15 (7+8 Kronrod) or 21 (10+11)

    Returns
    -------
    (estimate, error_estimate)
    """
    # 7-point Gauss nodes and weights
    g_nodes = jnp.array([
        -0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
        0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585,
    ])
    g_weights = jnp.array([
        0.1294849661688697, 0.2797053914892767, 0.3818300505051189,
        0.4179591836734694, 0.3818300505051189, 0.2797053914892767, 0.1294849661688697,
    ])

    # 15-point Kronrod nodes and weights
    k_nodes = jnp.array([
        -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
        -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
        -0.2077849550078985, 0.0,
        0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
        0.7415311855993945, 0.8648644233597691, 0.9491079123427585,
        0.9914553711208126,
    ])
    k_weights = jnp.array([
        0.0229353220105292, 0.0630920926299786, 0.1047900103222502,
        0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
        0.2044329400752989, 0.2094821410847278,
        0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
        0.1406532597155259, 0.1047900103222502, 0.0630920926299786,
        0.0229353220105292,
    ])

    half = (b - a) / 2.0
    mid = (a + b) / 2.0

    # Evaluate at Kronrod nodes
    x_k = mid + half * k_nodes
    f_k = jnp.array([f(xi) for xi in x_k])

    # Kronrod estimate
    I_k = half * jnp.sum(k_weights * f_k)

    # Gauss estimate (using shared nodes)
    x_g = mid + half * g_nodes
    f_g = jnp.array([f(xi) for xi in x_g])
    I_g = half * jnp.sum(g_weights * f_g)

    return I_k, jnp.abs(I_k - I_g)


def tanh_sinh_integral(f, a, b, n=50, h=0.1):
    """Tanh-sinh (double exponential) quadrature.

    Excellent for integrands with endpoint singularities.

    Parameters
    ----------
    f : callable
    a, b : bounds
    n : number of points per side
    h : step spacing

    Returns
    -------
    float
    """
    half = (b - a) / 2.0
    mid = (a + b) / 2.0

    result = 0.0
    for k in range(-n, n + 1):
        t = k * h
        # tanh-sinh transformation
        sinh_t = jnp.sinh(t)
        cosh_t = jnp.cosh(t)
        x = jnp.tanh(0.5 * jnp.pi * sinh_t)
        w = 0.5 * jnp.pi * cosh_t / jnp.cosh(0.5 * jnp.pi * sinh_t) ** 2
        s = mid + half * x
        # Skip if outside bounds (numerical)
        val = jnp.where((s > a) & (s < b), f(s), 0.0)
        result = result + h * w * val

    return half * result


def filon_integral(f, omega, a, b, n=100):
    """Filon quadrature for oscillatory integrals.

    Computes integral_a^b f(x) * cos(omega * x) dx.

    Parameters
    ----------
    f : callable (non-oscillatory part)
    omega : oscillation frequency
    a, b : bounds
    n : number of subintervals (must be even)

    Returns
    -------
    float
    """
    n = max(n, 2)
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    theta = omega * h
    theta2 = theta ** 2

    # Filon coefficients
    alpha = (theta2 + theta * jnp.sin(theta) * jnp.cos(theta) - 2.0 * jnp.sin(theta) ** 2) / theta2 ** (3.0 / 2.0) * jnp.where(jnp.abs(theta) > 1e-4, 1.0, 0.0) + jnp.where(jnp.abs(theta) <= 1e-4, 2.0 * theta / 45.0, 0.0)
    beta_f = 2.0 * (theta * (1 + jnp.cos(theta) ** 2) - 2 * jnp.sin(theta) * jnp.cos(theta)) / theta2 ** (3.0 / 2.0) * jnp.where(jnp.abs(theta) > 1e-4, 1.0, 0.0) + jnp.where(jnp.abs(theta) <= 1e-4, 2.0 / 3.0, 0.0)
    gamma_f = 4.0 * (jnp.sin(theta) - theta * jnp.cos(theta)) / theta2 ** (3.0 / 2.0) * jnp.where(jnp.abs(theta) > 1e-4, 1.0, 0.0) + jnp.where(jnp.abs(theta) <= 1e-4, 4.0 / 3.0, 0.0)

    x = a + h * jnp.arange(n + 1)
    fx = jnp.array([f(xi) for xi in x])

    # Sums
    C_even = jnp.sum(fx[0::2] * jnp.cos(omega * x[0::2]))
    C_odd = jnp.sum(fx[1::2] * jnp.cos(omega * x[1::2]))
    S_even = jnp.sum(fx[0::2] * jnp.sin(omega * x[0::2]))
    S_odd = jnp.sum(fx[1::2] * jnp.sin(omega * x[1::2]))

    return h * (alpha * (fx[-1] * jnp.sin(omega * b) - fx[0] * jnp.sin(omega * a))
                + beta_f * C_even + gamma_f * C_odd)


def two_dimensional_integral(f, a1, b1, a2, b2, n1=20, n2=20):
    """2D numerical integration using tensor-product Gauss-Legendre.

    Parameters
    ----------
    f : callable(x, y) -> float
    a1, b1 : bounds in first dimension
    a2, b2 : bounds in second dimension
    n1, n2 : quadrature points in each dimension

    Returns
    -------
    float
    """
    from ql_jax.math.integrals import gauss_legendre_nodes_weights

    nodes1, w1 = gauss_legendre_nodes_weights(n1)
    nodes2, w2 = gauss_legendre_nodes_weights(n2)

    # Transform from [-1,1] to [a,b]
    x1 = (b1 - a1) / 2.0 * nodes1 + (a1 + b1) / 2.0
    x2 = (b2 - a2) / 2.0 * nodes2 + (a2 + b2) / 2.0
    w1 = w1 * (b1 - a1) / 2.0
    w2 = w2 * (b2 - a2) / 2.0

    result = 0.0
    for i in range(n1):
        for j in range(n2):
            result = result + w1[i] * w2[j] * f(x1[i], x2[j])
    return result


def discrete_integral(x, y, method='trapezoid'):
    """Numerical integration of discrete data.

    Parameters
    ----------
    x : array of x values (must be sorted)
    y : array of y values
    method : 'trapezoid' or 'simpson'

    Returns
    -------
    float
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    if method == 'trapezoid':
        dx = jnp.diff(x)
        avg_y = 0.5 * (y[:-1] + y[1:])
        return jnp.sum(dx * avg_y)
    elif method == 'simpson':
        # Composite Simpson's rule (requires odd number of points)
        n = len(x) - 1
        dx = jnp.diff(x)
        result = 0.0
        for i in range(0, n - 1, 2):
            h = (x[i + 2] - x[i]) / 2.0
            result = result + h / 3.0 * (y[i] + 4 * y[i + 1] + y[i + 2])
        # If even number of intervals, add last trapezoid
        if n % 2 != 0:
            result = result + 0.5 * dx[-1] * (y[-2] + y[-1])
        return result
    else:
        raise ValueError(f"Unknown method: {method}")
