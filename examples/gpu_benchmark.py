"""Example: GPU vs CPU Performance Comparison.

Benchmarks QL-JAX operations on CPU vs GPU side-by-side:
  - Vectorized option pricing (vmap)
  - Monte Carlo simulation
  - AD Greeks computation

Requires a CUDA-enabled GPU and jaxlib[cuda] installed.
Falls back to CPU-only if GPU not available.
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


def _collect_devices():
    """Detect CPU and GPU devices."""
    devs = {"cpu": jax.devices("cpu")[0]}
    try:
        gpu_list = jax.devices("gpu")
        if gpu_list:
            devs["gpu"] = gpu_list[0]
    except RuntimeError:
        pass
    return devs


def main():
    devices = _collect_devices()
    has_gpu = "gpu" in devices

    print("=" * 60)
    print("QL-JAX Example: CPU vs GPU Performance Comparison")
    print("=" * 60)
    print(f"  Available devices: {list(devices.keys())}")
    if not has_gpu:
        print("  ⚠  No GPU detected — running CPU-only benchmarks")
    print()

    # ── Benchmark 1: Vectorized BS pricing ───────────────────
    n = 100_000

    bs_results = {}
    for label, dev in devices.items():
        spots = jax.device_put(jnp.full(n, 100.0), dev)
        strikes = jax.device_put(jnp.linspace(50.0, 150.0, n), dev)
        mats = jax.device_put(jnp.full(n, 1.0), dev)
        rates = jax.device_put(jnp.full(n, 0.05), dev)
        divs = jax.device_put(jnp.full(n, 0.02), dev)
        vols = jax.device_put(jnp.full(n, 0.20), dev)

        bs_vmap = jax.jit(jax.vmap(
            lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1)
        ))
        t_bs, _ = benchmark(bs_vmap, spots, strikes, mats, rates, divs, vols)
        bs_results[label] = t_bs

    print(f"  1. BS pricing {n:,d} options (vmap + jit):")
    for label, t_bs in bs_results.items():
        print(f"     [{label.upper()}] {t_bs*1000:.2f} ms  ({n/t_bs:,.0f} options/sec)")
    if has_gpu:
        spd = bs_results["cpu"] / bs_results["gpu"]
        print(f"     → GPU speedup: {spd:.1f}×")
    print()

    # ── Benchmark 2: Portfolio Greeks ────────────────────────
    greeks_results = {}
    for label, dev in devices.items():
        strikes = jax.device_put(jnp.linspace(50.0, 150.0, n), dev)

        delta_vmap = jax.jit(jax.vmap(
            jax.grad(lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1), argnums=0)
        ))
        gamma_vmap = jax.jit(jax.vmap(
            jax.grad(jax.grad(lambda s, k, t, r_, q_, v: black_scholes_price(s, k, t, r_, q_, v, 1), argnums=0), argnums=0)
        ))

        spots = jax.device_put(jnp.full(n, 100.0), dev)
        mats = jax.device_put(jnp.full(n, 1.0), dev)
        rates = jax.device_put(jnp.full(n, 0.05), dev)
        divs = jax.device_put(jnp.full(n, 0.02), dev)
        vols = jax.device_put(jnp.full(n, 0.20), dev)

        t_delta, _ = benchmark(delta_vmap, spots, strikes, mats, rates, divs, vols)
        t_gamma, _ = benchmark(gamma_vmap, spots, strikes, mats, rates, divs, vols)
        greeks_results[label] = (t_delta, t_gamma)

    print(f"  2. Portfolio Greeks ({n:,d} options):")
    for label, (t_d, t_g) in greeks_results.items():
        print(f"     [{label.upper()}] Delta: {t_d*1000:.2f} ms  Gamma: {t_g*1000:.2f} ms")
    if has_gpu:
        spd_d = greeks_results["cpu"][0] / greeks_results["gpu"][0]
        spd_g = greeks_results["cpu"][1] / greeks_results["gpu"][1]
        print(f"     → GPU speedup: Delta {spd_d:.1f}×  Gamma {spd_g:.1f}×")
    print()

    # ── Benchmark 3: Monte Carlo ─────────────────────────────
    key = jax.random.PRNGKey(42)
    n_paths_list = [10_000, 100_000]

    print(f"  3. Monte Carlo European option:")
    for n_paths in n_paths_list:
        mc_times = {}
        for label, dev in devices.items():
            k = jax.device_put(key, dev)
            mc_fn = jax.jit(lambda k: mc_european_bs(
                S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.20,
                option_type=1, n_paths=n_paths, n_steps=252, key=k,
            ))
            t_mc, mc_result = benchmark(mc_fn, k)
            mc_times[label] = t_mc
            mc_price = mc_result[0] if isinstance(mc_result, tuple) else mc_result

        for label, t_mc in mc_times.items():
            print(f"     [{label.upper()}] {n_paths:>10,d} paths: {t_mc*1000:.2f} ms")
        if has_gpu:
            spd = mc_times["cpu"] / mc_times["gpu"]
            print(f"     → GPU speedup ({n_paths:,d} paths): {spd:.1f}×")
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
