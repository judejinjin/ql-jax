"""Global bootstrap for yield curves.

Unlike sequential bootstrapping, global bootstrap solves for all
discount factors simultaneously by minimizing repricing errors.
"""

import jax
import jax.numpy as jnp


def global_bootstrap(helpers, initial_times, initial_rates=None,
                      max_iterations=100, tolerance=1e-10):
    """Globally calibrate a yield curve to reprice all helpers simultaneously.

    Parameters
    ----------
    helpers : list of rate helpers (each has .quote_error(discount_fn) method)
    initial_times : array – pillar maturities
    initial_rates : array – initial zero rates (default: 0.03)
    max_iterations : int
    tolerance : float

    Returns dict with: times, zero_rates, discount_factors.
    """
    n = len(initial_times)
    if initial_rates is None:
        initial_rates = jnp.full(n, 0.03)

    rates = jnp.asarray(initial_rates, dtype=jnp.float64)
    times = jnp.asarray(initial_times, dtype=jnp.float64)

    def make_discount_fn(zero_rates):
        def discount(t):
            return jnp.exp(-jnp.interp(t, times, zero_rates) * t)
        return discount

    def objective(zero_rates):
        discount_fn = make_discount_fn(zero_rates)
        errors = jnp.array([h.quote_error(discount_fn) for h in helpers])
        return jnp.sum(errors**2)

    # Optimize using L-BFGS
    from ql_jax.math.optimization.bfgs import minimize
    result = minimize(objective, rates, max_iterations=max_iterations, tol=tolerance)

    optimal_rates = result['x']
    return {
        'times': times,
        'zero_rates': optimal_rates,
        'discount_factors': jnp.exp(-optimal_rates * times),
        'converged': result.get('converged', True),
        'n_iterations': result.get('n_iterations', 0),
    }
