"""Benchmark: Analytic pricing engines (BSM, Heston).

Measures throughput at various batch sizes using jax.vmap.
"""

import time
import json
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price
from ql_jax.engines.analytic.heston import heston_price


def bench_bsm(n, n_warmup=3, n_timed=10):
    """Benchmark BSM pricing with vmap over n strikes."""
    S = 100.0
    T, r, q, sigma = 1.0, 0.05, 0.02, 0.20
    strikes = jnp.linspace(80.0, 120.0, n)

    price_fn = jax.vmap(
        lambda k: black_scholes_price(S, k, T, r, q, sigma, 1)
    )

    # Warmup
    for _ in range(n_warmup):
        price_fn(strikes).block_until_ready()

    # Timed
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        price_fn(strikes).block_until_ready()
        times.append(time.perf_counter() - t0)

    return {
        "engine": "BSM",
        "batch_size": n,
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "p5_ms": sorted(times)[0] * 1000,
        "p95_ms": sorted(times)[-1] * 1000,
        "throughput_per_sec": n / (sum(times) / len(times)),
    }


def bench_heston(n, n_warmup=3, n_timed=10):
    """Benchmark Heston pricing with vmap over n strikes."""
    S = 100.0
    T, r, q = 1.0, 0.05, 0.02
    v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.7
    strikes = jnp.linspace(80.0, 120.0, n)

    price_fn = jax.vmap(
        lambda k: heston_price(S, k, T, r, q, v0, kappa, theta, xi, rho, 1)
    )

    for _ in range(n_warmup):
        price_fn(strikes).block_until_ready()

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        price_fn(strikes).block_until_ready()
        times.append(time.perf_counter() - t0)

    return {
        "engine": "Heston",
        "batch_size": n,
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "p5_ms": sorted(times)[0] * 1000,
        "p95_ms": sorted(times)[-1] * 1000,
        "throughput_per_sec": n / (sum(times) / len(times)),
    }


def main():
    sizes = [10, 100, 1_000, 10_000, 100_000]
    results = []

    print("=" * 60)
    print("QL-JAX Benchmark: Analytic Engines")
    print("=" * 60)

    for n in sizes:
        r = bench_bsm(n)
        results.append(r)
        print(f"  BSM      n={n:>7d}  median={r['median_ms']:8.3f}ms  throughput={r['throughput_per_sec']:>12,.0f}/s")

    print()
    for n in sizes:
        r = bench_heston(n)
        results.append(r)
        print(f"  Heston   n={n:>7d}  median={r['median_ms']:8.3f}ms  throughput={r['throughput_per_sec']:>12,.0f}/s")

    return results


if __name__ == "__main__":
    main()
