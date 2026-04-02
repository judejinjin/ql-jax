"""Tree-based swaption and swap pricing engines."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.engines.lattice.short_rate_tree import (
    hw_trinomial_tree, hw_tree_swaption_price,
)


def tree_swaption_price(
    notional: float, fixed_rate: float,
    exercise_time: float,
    payment_dates: jnp.ndarray,
    day_fractions: jnp.ndarray,
    a: float, sigma: float,
    discount_fn,
    n_steps: int = 200,
    payer: bool = True,
) -> float:
    """Price a European swaption using Hull-White trinomial tree.

    Parameters
    ----------
    notional : notional amount
    fixed_rate : swap fixed rate
    exercise_time : swaption exercise date (years)
    payment_dates : swap payment dates (year fracs)
    day_fractions : day count fracs
    a, sigma : Hull-White parameters
    discount_fn : t -> discount factor
    n_steps : tree steps
    payer : True for payer swaption
    """
    T = float(payment_dates[-1])
    tree = hw_trinomial_tree(a, sigma, T, n_steps, discount_fn)

    # Find tree step closest to exercise
    dt = T / n_steps
    ex_step = int(round(exercise_time / dt))
    ex_step = max(1, min(ex_step, n_steps - 1))

    # Find steps for each payment
    pay_steps = []
    for t in payment_dates:
        step = int(round(float(t) / dt))
        step = min(step, n_steps)
        pay_steps.append(step)

    return hw_tree_swaption_price(
        tree, ex_step, pay_steps[-1],
        fixed_rate, notional, day_fractions,
        payer=payer,
    )


def tree_swap_price(
    notional: float, fixed_rate: float,
    payment_dates: jnp.ndarray,
    day_fractions: jnp.ndarray,
    a: float, sigma: float,
    discount_fn,
    n_steps: int = 200,
    payer: bool = True,
) -> float:
    """Price a swap using Hull-White trinomial tree.

    Parameters
    ----------
    notional : notional
    fixed_rate : fixed rate
    payment_dates : payment dates (year fracs)
    day_fractions : day count fracs
    a, sigma : Hull-White parameters
    discount_fn : discount curve
    n_steps : tree steps
    payer : True = pay fixed
    """
    T = float(payment_dates[-1])
    tree = hw_trinomial_tree(a, sigma, T, n_steps, discount_fn)
    dt = T / n_steps

    # Discount each cash flow
    total = 0.0
    for i in range(len(payment_dates)):
        t = float(payment_dates[i])
        df = discount_fn(t)
        tau = float(day_fractions[i])
        # Fixed leg
        fixed_cf = -fixed_rate * tau * notional
        # Float leg approximated at par
        if i == 0:
            t_prev = 0.0
        else:
            t_prev = float(payment_dates[i - 1])
        fwd = (discount_fn(t_prev) / discount_fn(t) - 1.0) / tau
        float_cf = fwd * tau * notional

        omega = 1.0 if payer else -1.0
        total = total + omega * (float_cf + fixed_cf) * df

    return total
