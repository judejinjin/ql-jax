"""Benchmark runner CLI for QL-JAX.

Usage:
    python runner.py --backend cpu --size large --precision float64
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="QL-JAX Benchmark Runner")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    args = parser.parse_args()

    print(f"QL-JAX Benchmark Runner")
    print(f"  Backend:   {args.backend}")
    print(f"  Size:      {args.size}")
    print(f"  Precision: {args.precision}")
    print(f"  (No benchmarks implemented yet — coming in Phase 8)")


if __name__ == "__main__":
    main()
