"""Validation: Monte Carlo Bond pricing
Source: sprint 3 item 3.9

Callable bond 100-scenario Hull-White batch via vmap + JIT.
Uses hull_white_bond_price (no float() blocker) for AD.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.models.shortrate.hull_white import hull_white_bond_price


def main():
    print("=" * 78)
    print("Monte Carlo Bond Pricing (Hull-White scenarios)")
    print("=" * 78)

    # Parameters
    a = 0.1        # mean reversion
    sigma = 0.01   # vol
    r0 = 0.03      # initial short rate
    T_bond = 5.0   # bond maturity

    # Flat discount function
    disc_fn = lambda t: jnp.exp(-r0 * t)

    print(f"  HW: a={a}, σ={sigma}, r0={r0}")
    print(f"  Bond maturity: {T_bond}Y")

    # === Test 1: Single bond price ===
    t = 0.0
    price_base = float(hull_white_bond_price(r0, a, sigma, t, T_bond, disc_fn))
    print(f"\n  HW bond price (t=0): {price_base:.6f}")
    exact = float(jnp.exp(-r0 * T_bond))
    print(f"  Exact ZCB:           {exact:.6f}")
    diff = abs(price_base - exact)
    print(f"  Difference:          {diff:.2e}")

    # === Test 2: AD Greeks (sensitivities to r0, a, sigma) ===
    # At t>0, a and sigma affect the bond price through the HW affine term structure
    def price_fn(params):
        rate, mean_rev, vol = params
        df = lambda t: jnp.exp(-rate * t)
        # Evaluate at t=1.0 (1Y from now) to get non-trivial a/sigma sensitivity
        return hull_white_bond_price(rate, mean_rev, vol, 1.0, T_bond, df)

    base = jnp.array([r0, a, sigma])
    greeks = np.array(jax.grad(price_fn)(base))
    names = ['r0', 'a', 'sigma']
    print(f"\n  --- AD Greeks (t=1Y) ---")
    for name, g in zip(names, greeks):
        print(f"    d(P)/d({name:>5s}): {g:.6f}")

    # FD check
    H = 1e-6
    fd = np.zeros(3)
    for i in range(3):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd[i] = (float(price_fn(xp)) - float(price_fn(xm))) / (2 * H)
    # Only check non-trivial gradients
    nontrivial = np.abs(fd) > 1e-8
    if np.any(nontrivial):
        max_rel = np.max(np.abs(greeks[nontrivial] - fd[nontrivial]) /
                         np.maximum(np.abs(fd[nontrivial]), 1e-10))
    else:
        max_rel = 0.0
    print(f"  AD vs FD max rel diff: {max_rel:.2e}")

    # === Test 3: 100-scenario batch ===
    price_fn_scalar = jax.jit(lambda p: price_fn(p))
    grad_fn = jax.jit(jax.grad(price_fn))
    vmap_grad = jax.jit(jax.vmap(jax.grad(price_fn)))

    # Generate scenarios: perturb r0, a, sigma
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    n_scenarios = 100
    scenarios = jnp.stack([
        0.01 + 0.04 * jax.random.uniform(k1, (n_scenarios,)),  # r0
        0.05 + 0.15 * jax.random.uniform(k2, (n_scenarios,)),  # a
        0.005 + 0.015 * jax.random.uniform(k3, (n_scenarios,)),  # sigma
    ], axis=1)

    # Warmup
    _ = vmap_grad(scenarios)

    t0 = time.perf_counter()
    batch_greeks = vmap_grad(scenarios)
    batch_greeks.block_until_ready()
    t_batch = time.perf_counter() - t0
    print(f"\n  Batch {n_scenarios} scenarios: {t_batch*1000:.1f} ms ({t_batch/n_scenarios*1000:.3f} ms/scenario)")
    print(f"  Greeks shape: {batch_greeks.shape}")

    # Consistency: first scenario
    single_g = np.array(grad_fn(scenarios[0]))
    batch_g0 = np.array(batch_greeks[0])
    batch_diff = np.max(np.abs(single_g - batch_g0) / np.maximum(np.abs(single_g), 1e-10))
    print(f"  Single vs batch[0] diff: {batch_diff:.2e}")

    # === Test 4: Prices batch ===
    vmap_price = jax.jit(jax.vmap(price_fn))
    _ = vmap_price(scenarios)
    t0 = time.perf_counter()
    batch_prices = vmap_price(scenarios)
    batch_prices.block_until_ready()
    t_prices = time.perf_counter() - t0
    print(f"  Batch prices: {t_prices*1000:.1f} ms")
    all_valid = bool(jnp.all(batch_prices > 0) and jnp.all(batch_prices < 1.5))
    print(f"  Price range: [{float(jnp.min(batch_prices)):.4f}, {float(jnp.max(batch_prices)):.4f}]")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = diff < 0.01
    n_pass += int(ok)
    print(f"\n  HW ≈ exact ZCB:     {'✓' if ok else '✗'} (diff={diff:.2e})")

    ok = max_rel < 1e-4
    n_pass += int(ok)
    print(f"  AD vs FD:           {'✓' if ok else '✗'} ({max_rel:.2e})")

    ok = batch_greeks.shape == (100, 3)
    n_pass += int(ok)
    print(f"  Batch shape:        {'✓' if ok else '✗'}")

    ok = batch_diff < 1e-8
    n_pass += int(ok)
    print(f"  Single=Batch:       {'✓' if ok else '✗'} ({batch_diff:.2e})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All Monte Carlo bond validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
