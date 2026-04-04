"""Benchmark: Analytic pricing engines (BSM, Heston).

Measures throughput at various batch sizes using jax.vmap.
Supports explicit device placement for CPU vs GPU comparison.
"""

import time
import json
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price
from ql_jax.engines.analytic.heston import heston_price


def _place(arr, device):
    """Place array on a specific device, or leave on default."""
    if device is None:
        return arr
    return jax.device_put(arr, device)


def bench_bsm(n, device=None, n_warmup=3, n_timed=10):
    """Benchmark BSM pricing with vmap over n strikes.

    Parameters
    ----------
    device : jax.Device or None
        Target device (e.g. ``jax.devices('cpu')[0]``).  None uses default.
    """
    S = 100.0
    T, r, q, sigma = 1.0, 0.05, 0.02, 0.20
    strikes = _place(jnp.linspace(80.0, 120.0, n), device)

    @jax.jit
    def price_fn(ks):
        return jax.vmap(lambda k: black_scholes_price(S, k, T, r, q, sigma, 1))(ks)

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
        "device": str(device) if device else str(jax.default_backend()),
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "p5_ms": sorted(times)[0] * 1000,
        "p95_ms": sorted(times)[-1] * 1000,
        "throughput_per_sec": n / (sum(times) / len(times)),
    }


def bench_heston(n, device=None, n_warmup=3, n_timed=10):
    """Benchmark Heston pricing with vmap over n strikes.

    Parameters
    ----------
    device : jax.Device or None
        Target device.  None uses default.
    """
    S = 100.0
    T, r, q = 1.0, 0.05, 0.02
    v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.7
    strikes = _place(jnp.linspace(80.0, 120.0, n), device)

    @jax.jit
    def price_fn(ks):
        return jax.vmap(
            lambda k: heston_price(S, k, T, r, q, v0, kappa, theta, xi, rho, 1)
        )(ks)

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
        "device": str(device) if device else str(jax.default_backend()),
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "p5_ms": sorted(times)[0] * 1000,
        "p95_ms": sorted(times)[-1] * 1000,
        "throughput_per_sec": n / (sum(times) / len(times)),
    }


def _collect_devices():
    """Return a dict of {label: jax.Device} for available backends."""
    devs = {}
    devs["cpu"] = jax.devices("cpu")[0]
    try:
        gpu_list = jax.devices("gpu")
        if gpu_list:
            devs["gpu"] = gpu_list[0]
    except RuntimeError:
        pass
    return devs


def main(devices=None):
    """Run analytic benchmarks on one or more devices.

    Parameters
    ----------
    devices : dict[str, jax.Device] or None
        Map of label -> device.  None auto-detects.
    """
    if devices is None:
        devices = _collect_devices()

    sizes = [10, 100, 1_000, 10_000, 100_000]
    results = []

    print("=" * 60)
    print("QL-JAX Benchmark: Analytic Engines")
    print(f"  Devices: {list(devices.keys())}")
    print("=" * 60)

    for label, dev in devices.items():
        print(f"\n  [{label.upper()}]")
        for n in sizes:
            r = bench_bsm(n, device=dev)
            results.append(r)
            print(f"    BSM      n={n:>7d}  median={r['median_ms']:8.3f}ms  "
                  f"throughput={r['throughput_per_sec']:>12,.0f}/s")

        for n in sizes:
            r = bench_heston(n, device=dev)
            results.append(r)
            print(f"    Heston   n={n:>7d}  median={r['median_ms']:8.3f}ms  "
                  f"throughput={r['throughput_per_sec']:>12,.0f}/s")

    # Print comparison table if both CPU and GPU results exist
    if "cpu" in devices and "gpu" in devices:
        print("\n  ── CPU vs GPU Speedup ─────────────────────────")
        cpu_map = {(r["engine"], r["batch_size"]): r
                   for r in results if "cpu" in r["device"].lower()}
        gpu_map = {(r["engine"], r["batch_size"]): r
                   for r in results if "gpu" in r["device"].lower()}
        print(f"    {'Engine':<8} {'N':>8}  {'CPU ms':>9}  {'GPU ms':>9}  {'Speedup':>8}")
        for key in sorted(cpu_map):
            if key in gpu_map:
                c = cpu_map[key]["median_ms"]
                g = gpu_map[key]["median_ms"]
                spd = c / g if g > 0 else float("inf")
                print(f"    {key[0]:<8} {key[1]:>8d}  {c:>9.3f}  {g:>9.3f}  {spd:>7.1f}x")

    return results


if __name__ == "__main__":
    main()
