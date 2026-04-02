"""Short-rate trinomial lattice – Hull-White / BK tree for IR derivatives."""

import jax.numpy as jnp
import jax


def hw_trinomial_tree(a, sigma, T, n_steps, discount_fn):
    """Build a Hull-White trinomial tree.

    Returns tree structure (rates, probabilities) for pricing.

    Parameters
    ----------
    a : float – mean reversion speed
    sigma : float – short-rate vol
    T : float – maturity
    n_steps : int
    discount_fn : callable(t) -> DF

    Returns
    -------
    dict with keys:
        'dt' : float
        'dx' : float
        'rates' : list of arrays (rate levels at each step)
        'probs' : list of (p_u, p_m, p_d) arrays
        'alpha' : array – calibrated drift adjustments
    """
    dt = T / n_steps
    dx = sigma * jnp.sqrt(3.0 * dt)
    j_max = int(jnp.ceil(0.184 / (a * dt)))  # Standard truncation

    # Build state space and probabilities
    rates = []
    probs = []
    alphas = jnp.zeros(n_steps + 1)

    for step in range(n_steps + 1):
        n_nodes = min(2 * step + 1, 2 * j_max + 1)
        j_range = jnp.arange(-(n_nodes // 2), n_nodes // 2 + 1)
        x_vals = j_range * dx

        if step == 0:
            # Calibrate alpha[0] to match market discount
            target_df = discount_fn(dt)
            # alpha[0] = -log(target_df)/dt approximately
            alpha_0 = -jnp.log(target_df) / dt
            alphas = alphas.at[0].set(alpha_0)
            rates.append(alpha_0 + x_vals)
        else:
            # Forward induction to calibrate alpha
            # Simple approach: match expected discount factor
            target_df = discount_fn((step + 1) * dt) / discount_fn(step * dt)
            alpha_step = -jnp.log(target_df) / dt
            alphas = alphas.at[step].set(alpha_step)
            rates.append(alpha_step + x_vals)

        # Transition probabilities
        if step < n_steps:
            eta = a * j_range * dx * dt
            p_u = 1.0 / 6.0 + (eta**2 - eta) / (2.0 * dx**2 / dt)
            p_m = 2.0 / 3.0 - eta**2 / (dx**2 / dt)
            p_d = 1.0 / 6.0 + (eta**2 + eta) / (2.0 * dx**2 / dt)

            # Standard HW probabilities
            p_u = 1.0 / 6.0 + 0.5 * (a**2 * j_range**2 * dt - a * j_range * dt)
            p_m = 2.0 / 3.0 - a**2 * j_range**2 * dt
            p_d = 1.0 / 6.0 + 0.5 * (a**2 * j_range**2 * dt + a * j_range * dt)

            probs.append((p_u, p_m, p_d))

    return {
        'dt': dt, 'dx': dx, 'rates': rates, 'probs': probs,
        'alpha': alphas, 'j_max': j_max,
    }


def hw_tree_bond_price(tree, maturity_step):
    """Price a zero-coupon bond on the HW tree via backward induction.

    Parameters
    ----------
    tree : dict from hw_trinomial_tree
    maturity_step : int – step index for bond maturity

    Returns
    -------
    price : float
    """
    dt = tree['dt']
    rates = tree['rates']
    probs = tree['probs']

    # Terminal: bond pays 1
    n_terminal = len(rates[maturity_step])
    values = jnp.ones(n_terminal)

    for step in range(maturity_step - 1, -1, -1):
        r_step = rates[step]
        p_u, p_m, p_d = probs[step]

        n_curr = len(r_step)
        disc = jnp.exp(-r_step * dt)

        new_values = jnp.zeros(n_curr)
        for j in range(n_curr):
            # Map to next step indices (with branching)
            j_next = j  # center maps to same j in next level
            v_u = values[min(j_next + 1, len(values) - 1)]
            v_m = values[min(j_next, len(values) - 1)]
            v_d = values[max(j_next - 1, 0)]

            new_values = new_values.at[j].set(
                disc[j] * (p_u[j] * v_u + p_m[j] * v_m + p_d[j] * v_d)
            )
        values = new_values

    return values[len(values) // 2]


def hw_tree_swaption_price(tree, swap_start_step, swap_end_step,
                             fixed_rate, notional=1.0, is_payer=True):
    """Price a European swaption on the HW tree.

    Parameters
    ----------
    tree : dict from hw_trinomial_tree
    swap_start_step : int – step at which swap starts
    swap_end_step : int – step at which swap ends
    fixed_rate : float
    notional : float
    is_payer : bool

    Returns
    -------
    price : float
    """
    dt = tree['dt']
    rates = tree['rates']
    probs = tree['probs']

    # At swap_start_step, compute swap value at each node
    n_nodes = len(rates[swap_start_step])

    # Approximate: swap value ≈ notional * (1 - P(T_start, T_end) - fixed_rate * annuity)
    # For each node, compute zero-coupon bond to swap end
    swap_values = jnp.zeros(n_nodes)
    for j in range(n_nodes):
        # Approximate bond price using short rate at this node
        r_j = rates[swap_start_step][j]
        tau = (swap_end_step - swap_start_step) * dt
        # Simple approximation: P ≈ exp(-r * tau)
        bond_price = jnp.exp(-r_j * tau)
        annuity = (1.0 - bond_price) / r_j  # Approximate annuity
        swap_val = notional * (1.0 - bond_price - fixed_rate * annuity)
        if not is_payer:
            swap_val = -swap_val
        swap_values = swap_values.at[j].set(jnp.maximum(swap_val, 0.0))

    # Backward induction to time 0
    values = swap_values
    for step in range(swap_start_step - 1, -1, -1):
        r_step = rates[step]
        p_u, p_m, p_d = probs[step]
        n_curr = len(r_step)
        disc = jnp.exp(-r_step * dt)

        new_values = jnp.zeros(n_curr)
        for j in range(n_curr):
            j_next = j
            v_u = values[min(j_next + 1, len(values) - 1)]
            v_m = values[min(j_next, len(values) - 1)]
            v_d = values[max(j_next - 1, 0)]
            new_values = new_values.at[j].set(
                disc[j] * (p_u[j] * v_u + p_m[j] * v_m + p_d[j] * v_d)
            )
        values = new_values

    return values[len(values) // 2]
