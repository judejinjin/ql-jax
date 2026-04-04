"""Benchmark runner CLI for QL-JAX.

Usage:
    python runner.py                        # auto-detect devices, run both if GPU present
    python runner.py --backend cpu          # CPU only
    python runner.py --backend gpu          # GPU only
    python runner.py --backend both         # explicit CPU + GPU comparison
    python runner.py --size large --precision float64
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
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    try:
        gpu_devs = jax.devices("gpu")
        info["gpu_devices"] = [str(d) for d in gpu_devs]
    except RuntimeError:
        info["gpu_devices"] = []
    return info


def _resolve_devices(backend_arg):
    """Turn the --backend CLI argument into a dict of {label: jax.Device}.

    'auto' detects what is available: CPU always, GPU if present.
    'both' requires GPU or raises an error.
    """
    cpu = jax.devices("cpu")[0]
    try:
        gpu_list = jax.devices("gpu")
        gpu = gpu_list[0] if gpu_list else None
    except RuntimeError:
        gpu = None

    if backend_arg == "cpu":
        return {"cpu": cpu}
    elif backend_arg == "gpu":
        if gpu is None:
            print("ERROR: --backend gpu requested but no GPU found. Falling back to CPU.")
            return {"cpu": cpu}
        return {"gpu": gpu}
    elif backend_arg == "both":
        if gpu is None:
            print("WARNING: --backend both requested but no GPU found. Running CPU only.")
            return {"cpu": cpu}
        return {"cpu": cpu, "gpu": gpu}
    else:  # auto
        devs = {"cpu": cpu}
        if gpu is not None:
            devs["gpu"] = gpu
        return devs


def main():
    parser = argparse.ArgumentParser(description="QL-JAX Benchmark Runner")
    parser.add_argument("--backend", choices=["cpu", "gpu", "both", "auto"],
                        default="auto",
                        help="Which device(s) to benchmark (default: auto-detect)")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    info = get_system_info()
    devices = _resolve_devices(args.backend)

    print(f"QL-JAX Benchmark Runner")
    print(f"  Backend:   {args.backend} → {list(devices.keys())}")
    print(f"  Size:      {args.size}")
    print(f"  Precision: {args.precision}")
    print(f"  JAX:       {info['jax_version']}")
    print(f"  CPU:       {info['jax_devices']}")
    if info['gpu_devices']:
        print(f"  GPU:       {info['gpu_devices']}")
    print()

    all_results = {"system": info, "config": vars(args), "benchmarks": []}

    # Run analytic benchmarks
    print("Running analytic benchmarks...")
    from benchmarks.bench_analytic import main as bench_analytic_main
    results = bench_analytic_main(devices=devices)
    all_results["benchmarks"].extend(results)
    print()

    # Run greeks benchmarks
    print("Running Greeks benchmarks...")
    from benchmarks.bench_greeks import main as bench_greeks_main
    bench_greeks_main(devices=devices)
    print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results written to {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
