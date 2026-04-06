"""Validation: Global Optimizer
Source: ~/QuantLib/Examples/GlobalOptimizer/
Tests DE, SA, BFGS on standard test functions (Rosenbrock, Ackley).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.math.optimization.differential_evolution import minimize as de_minimize
from ql_jax.math.optimization.simulated_annealing import minimize as sa_minimize
from ql_jax.math.optimization.bfgs import minimize as bfgs_minimize


def rosenbrock(x):
    """Rosenbrock function: min at (1,1), value 0."""
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


def sphere(x):
    """Sphere function: min at origin, value 0."""
    return jnp.sum(x**2)


def rastrigin(x):
    """Rastrigin function: min at origin, value 0."""
    n = x.shape[0]
    return 10.0 * n + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))


def main():
    print("=" * 78)
    print("Global Optimizer Benchmarks")
    print("=" * 78)

    n_pass = 0
    total = 0

    # === Test 1: BFGS on Rosenbrock ===
    print(f"\n  --- Rosenbrock (BFGS) ---")
    x0 = jnp.array([-1.0, 1.0])
    result = bfgs_minimize(rosenbrock, x0, max_iter=200, tol=1e-12)
    x_opt = result['x']
    f_opt = float(rosenbrock(x_opt))
    print(f"    x* = ({float(x_opt[0]):.6f}, {float(x_opt[1]):.6f})")
    print(f"    f* = {f_opt:.2e}")
    ok = f_opt < 1e-6 and abs(float(x_opt[0]) - 1.0) < 1e-3
    n_pass += int(ok)
    total += 1
    print(f"    {'✓' if ok else '✗'}")

    # === Test 2: DE on Rosenbrock ===
    print(f"\n  --- Rosenbrock (DE) ---")
    bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
    try:
        result = de_minimize(rosenbrock, bounds, max_iterations=500, population_size=40)
        x_opt = result['x']
        f_opt = float(rosenbrock(x_opt))
        print(f"    x* = ({float(x_opt[0]):.6f}, {float(x_opt[1]):.6f})")
        print(f"    f* = {f_opt:.2e}")
        ok = f_opt < 0.1
    except Exception as e:
        print(f"    DE: {e}")
        f_opt = 999.0
        ok = False
    n_pass += int(ok)
    total += 1
    print(f"    {'✓' if ok else '✗'}")

    # === Test 3: SA on Sphere ===
    print(f"\n  --- Sphere (SA) ---")
    x0 = jnp.array([5.0, -3.0, 2.0])
    try:
        result = sa_minimize(sphere, x0, max_iterations=5000)
        x_opt = result['x']
        f_opt = float(sphere(x_opt))
        print(f"    x* = ({', '.join(f'{float(v):.4f}' for v in x_opt)})")
        print(f"    f* = {f_opt:.2e}")
        ok = f_opt < 1.0
    except Exception as e:
        print(f"    SA: {e}")
        f_opt = 999.0
        ok = False
    n_pass += int(ok)
    total += 1
    print(f"    {'✓' if ok else '✗'}")

    # === Test 4: BFGS on Sphere ===
    print(f"\n  --- Sphere (BFGS) ---")
    x0 = jnp.array([5.0, -3.0, 2.0])
    result = bfgs_minimize(sphere, x0, max_iter=100, tol=1e-12)
    x_opt = result['x']
    f_opt = float(sphere(x_opt))
    print(f"    x* = ({', '.join(f'{float(v):.6f}' for v in x_opt)})")
    print(f"    f* = {f_opt:.2e}")
    ok = f_opt < 1e-10
    n_pass += int(ok)
    total += 1
    print(f"    {'✓' if ok else '✗'}")

    # === Test 5: JAX AD gradient at optimum ===
    print(f"\n  --- JAX AD gradient check ---")
    x_opt = jnp.array([1.0, 1.0])
    grad = np.array(jax.grad(rosenbrock)(x_opt))
    grad_norm = float(np.linalg.norm(grad))
    print(f"    ∇f at (1,1) = ({grad[0]:.2e}, {grad[1]:.2e}), |∇f| = {grad_norm:.2e}")
    ok = grad_norm < 1e-8
    n_pass += int(ok)
    total += 1
    print(f"    {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All optimizer validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
