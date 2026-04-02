"""Example: GPU vs CPU Performance Comparison.

Benchmarks QL-JAX operations on CPU vs GPU:
  - Vectorized option pricing (vmap)
  - Monte Carlo simulation
  - AD Greeks computation

Note: requires a CUDA-enabled GPU and jaxlib[cuda] installed.
Falls back to CPU-only comparison if GPU not available.
"""

import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price
from ql_jax.engines.mc.european import mc_european_bs


def benchmark(fn, *args, warmup=2, runs=10, label=""):
    """Run fn with warmup, then measure average time."""
    for _ in range(warmup):
        result = fn(*args)
        jax.block_until_ready(result)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    return avg, result


def main():
    devices = jax.devices()
    has_gpu = any(d.platform == 'gpu' for d in devices)

    print("=" * 60)
    print("QL-JAX Example: Performance Benchmarks")
    print("=" * 60)
    print(f"  Available devices: {[str(d) for d in devices]}")
    print(f"  GPU available: {has_gpu}")
    print()

    # ── Benchmark 1: Vectorized BS pricing ───────────────────
    n = 100_000
    spots = jnp.full(n, 100.0)
    strikes = jnp.linspace(50.0, 150.0, n)
    mats = jnp.full(n, 1.0)
    rates = jnp.full(n, 0.05)
    divs = jnp.full(n, 0.02)
    vols = jnp.full(n, 0.20)

    bs_vmap = jax.jit(jax.vmap(
        lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1)
    ))
    t_bs, _ = benchmark(bs_vmap, spots, strikes, mats, rates, divs, vols,
                         label="BS pricing (100k options)")
    print(f"  1. BS pricing {n:,d} options (vmap + jit):")
    print(f"     Time: {t_bs*1000:.2f} ms  ({n/t_bs:.0f} options/sec)")
    print()

    # ── Benchmark 2: Portfolio Greeks ────────────────────────
    delta_vmap = jax.jit(jax.vmap(
        jax.grad(lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1), argnums=0)
    ))
    gamma_vmap = jax.jit(jax.vmap(
        jax.grad(jax.grad(lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1), argnums=0), argnums=0)
    ))

    t_delta, _ = benchmark(delta_vmap, spots, strikes, mats, rates, divs, vols,
                            label="Deltas")
    t_gamma, _ = benchmark(gamma_vmap, spots, strikes, mats, rates, divs, vols,
                            label="Gammas")
    print(f"  2. Portfolio Greeks ({n:,d} options):")
    print(f"     Delta: {t_delta*1000:.2f} ms")
    print(f"     Gamma: {t_gamma*1000:.2f} ms")
    print()

    # ── Benchmark 3: Monte Carlo ─────────────────────────────
    key = jax.random.PRNGKey(42)
    n_paths_list = [10_000, 100_000]

    print(f"  3. Monte Carlo European option:")
    for n_paths in n_paths_list:
        mc_fn = jax.jit(lambda k: mc_european_bs(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.20,
            option_type=1, n_paths=n_paths, n_steps=252, key=k,
        ))
        t_mc, mc_result = benchmark(mc_fn, key, label=f"MC {n_paths}")
        mc_price = mc_result[0] if isinstance(mc_result, tuple) else mc_result
        print(f"     {n_paths:>10,d} paths: {t_mc*1000:.2f} ms  (price={float(mc_price):.4f})")
    print()

    # ── Benchmark 4: JIT compilation time ────────────────────
    print(f"  4. JIT compilation overhead:")

    @jax.jit
    def fresh_fn(S, K, T, r, q, sigma):
        return black_scholes_price(S, K, T, r, q, sigma, 1)

    start = time.perf_counter()
    result = fresh_fn(100.0, 100.0, 1.0, 0.05, 0.02, 0.20)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - start

    start = time.perf_counter()
    result = fresh_fn(100.0, 100.0, 1.0, 0.05, 0.02, 0.20)
    jax.block_until_ready(result)
    exec_time = time.perf_counter() - start

    print(f"     First call (compile + run): {compile_time*1000:.2f} ms")
    print(f"     Second call (cached):       {exec_time*1000:.4f} ms")
    print(f"     Speedup:                    {compile_time/exec_time:.0f}×")


if __name__ == "__main__":
    main()
