"""Benchmark runner CLI for QL-JAX.

Usage:
    python runner.py --backend cpu --size large --precision float64
"""

import argparse
import json
import os
import platform
import sys
import time

import jax
import jax.numpy as jnp


def get_system_info():
    """Collect system/hardware info."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(d) for d in jax.devices()],
    }


def main():
    parser = argparse.ArgumentParser(description="QL-JAX Benchmark Runner")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    info = get_system_info()
    print(f"QL-JAX Benchmark Runner")
    print(f"  Backend:   {args.backend} (active: {info['jax_backend']})")
    print(f"  Size:      {args.size}")
    print(f"  Precision: {args.precision}")
    print(f"  JAX:       {info['jax_version']}")
    print(f"  Devices:   {info['jax_devices']}")
    print()

    all_results = {"system": info, "config": vars(args), "benchmarks": []}

    # Run analytic benchmarks
    print("Running analytic benchmarks...")
    from benchmarks.bench_analytic import main as bench_analytic_main
    results = bench_analytic_main()
    all_results["benchmarks"].extend(results)
    print()

    # Run greeks benchmarks
    print("Running Greeks benchmarks...")
    from benchmarks.bench_greeks import main as bench_greeks_main
    bench_greeks_main()
    print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results written to {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
