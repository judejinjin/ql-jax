"""Benchmark: AD Greeks computation.

Measures the cost of computing Greeks via jax.grad vs finite difference.
Supports explicit device placement for CPU vs GPU comparison.
"""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price


def _place(arr, device):
    """Place array on a specific device, or leave on default."""
    if device is None:
        return arr
    return jax.device_put(arr, device)


def bench_greeks(n_strikes, device=None, n_warmup=3, n_timed=10):
    """Benchmark AD Greeks via grad + vmap over n strikes.

    Parameters
    ----------
    device : jax.Device or None
        Target device.  None uses default.
    """
    K_arr = _place(jnp.linspace(80.0, 120.0, n_strikes), device)
    T, r, q, sigma = 1.0, 0.05, 0.02, 0.20

    # Build the gradient function
    def delta_and_vega(S, K):
        delta = jax.grad(lambda s: black_scholes_price(s, K, T, r, q, sigma, 1))(S)
        vega = jax.grad(lambda sig: black_scholes_price(S, K, T, r, q, sig, 1))(sigma)
        return delta, vega

    batch_fn = jax.jit(jax.vmap(lambda k: delta_and_vega(100.0, k)))

    for _ in range(n_warmup):
        d, v = batch_fn(K_arr)
        d.block_until_ready()

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        d, v = batch_fn(K_arr)
        d.block_until_ready()
        v.block_until_ready()
        times.append(time.perf_counter() - t0)

    return {
        "method": "AD (grad+vmap)",
        "n_strikes": n_strikes,
        "device": str(device) if device else str(jax.default_backend()),
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "throughput_per_sec": n_strikes / (sum(times) / len(times)),
    }


def bench_fd_greeks(n_strikes, device=None, n_warmup=3, n_timed=10):
    """Benchmark finite-difference Greeks for comparison.

    Parameters
    ----------
    device : jax.Device or None
        Target device.  None uses default.
    """
    K_arr = _place(jnp.linspace(80.0, 120.0, n_strikes), device)
    T, r, q, sigma = 1.0, 0.05, 0.02, 0.20
    eps = 0.01

    def fd_delta_vega(S, K):
        p_up = black_scholes_price(S + eps, K, T, r, q, sigma, 1)
        p_dn = black_scholes_price(S - eps, K, T, r, q, sigma, 1)
        delta = (p_up - p_dn) / (2 * eps)

        p_up_v = black_scholes_price(S, K, T, r, q, sigma + eps, 1)
        p_dn_v = black_scholes_price(S, K, T, r, q, sigma - eps, 1)
        vega = (p_up_v - p_dn_v) / (2 * eps)
        return delta, vega

    batch_fn = jax.jit(jax.vmap(lambda k: fd_delta_vega(100.0, k)))

    for _ in range(n_warmup):
        d, v = batch_fn(K_arr)
        d.block_until_ready()

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        d, v = batch_fn(K_arr)
        d.block_until_ready()
        v.block_until_ready()
        times.append(time.perf_counter() - t0)

    return {
        "method": "FD (central)",
        "n_strikes": n_strikes,
        "device": str(device) if device else str(jax.default_backend()),
        "median_ms": sorted(times)[len(times) // 2] * 1000,
        "throughput_per_sec": n_strikes / (sum(times) / len(times)),
    }


def _collect_devices():
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
    """Run Greeks benchmarks on one or more devices.

    Parameters
    ----------
    devices : dict[str, jax.Device] or None
        Map of label -> device.  None auto-detects.
    """
    if devices is None:
        devices = _collect_devices()

    sizes = [100, 1_000, 10_000, 100_000]

    print("=" * 60)
    print("QL-JAX Benchmark: AD Greeks vs Finite Difference")
    print(f"  Devices: {list(devices.keys())}")
    print("=" * 60)

    for label, dev in devices.items():
        print(f"\n  [{label.upper()}]")
        for n in sizes:
            ad = bench_greeks(n, device=dev)
            fd = bench_fd_greeks(n, device=dev)
            print(f"    n={n:>7d}  AD: {ad['median_ms']:8.3f}ms  FD: {fd['median_ms']:8.3f}ms  "
                  f"ratio: {fd['median_ms']/max(ad['median_ms'],0.001):.1f}x")

    # Cross-device comparison for AD Greeks
    if "cpu" in devices and "gpu" in devices:
        print("\n  ── AD Greeks: CPU vs GPU Speedup ────────")
        print(f"    {'N':>8}  {'CPU ms':>9}  {'GPU ms':>9}  {'Speedup':>8}")
        for n in sizes:
            c = bench_greeks(n, device=devices["cpu"])
            g = bench_greeks(n, device=devices["gpu"])
            spd = c["median_ms"] / g["median_ms"] if g["median_ms"] > 0 else float("inf")
            print(f"    {n:>8d}  {c['median_ms']:>9.3f}  {g['median_ms']:>9.3f}  {spd:>7.1f}x")


if __name__ == "__main__":
    main()
