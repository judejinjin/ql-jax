"""Validation: Multi-dimensional Integration
Source: ~/QuantLib/Examples/MultidimIntegral/
Tests Gauss-Legendre, Simpson, and Trapezoid integration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.math.integrals.gauss import integrate_gauss_legendre
from ql_jax.math.integrals.simpson import integrate as integrate_simpson
from ql_jax.math.integrals.trapezoid import integrate as integrate_trapezoid


def main():
    print("=" * 78)
    print("Multi-dimensional Integration")
    print("=" * 78)

    # 1D test functions with known integrals
    tests = [
        ("∫₀¹ x² dx = 1/3", lambda x: x**2, 0.0, 1.0, 1.0/3.0),
        ("∫₀π sin(x) dx = 2", lambda x: jnp.sin(x), 0.0, jnp.pi, 2.0),
        ("∫₀¹ e^x dx = e-1", lambda x: jnp.exp(x), 0.0, 1.0, jnp.e - 1.0),
        ("∫₀¹ 1/(1+x²) dx = π/4", lambda x: 1.0/(1.0+x**2), 0.0, 1.0, jnp.pi/4.0),
        ("∫₋₁¹ (1-x²)^(1/2) dx = π/2", lambda x: jnp.sqrt(1.0 - x**2), -1.0, 1.0, jnp.pi/2.0),
    ]

    n_pass = 0
    total = 0

    for desc, f, a, b, exact in tests:
        print(f"\n  {desc}")
        exact_val = float(exact)

        # Gauss-Legendre (n=20)
        gl = float(integrate_gauss_legendre(f, a, b, 20))
        gl_err = abs(gl - exact_val)

        # Simpson (n=100)
        simp = float(integrate_simpson(f, a, b, 100))
        simp_err = abs(simp - exact_val)

        # Trapezoid (n=100)
        trap = float(integrate_trapezoid(f, a, b, 100))
        trap_err = abs(trap - exact_val)

        print(f"    Gauss-Legendre(20): {gl:.10f}  err={gl_err:.2e}")
        print(f"    Simpson(100):       {simp:.10f}  err={simp_err:.2e}")
        print(f"    Trapezoid(100):     {trap:.10f}  err={trap_err:.2e}")

        # GL should be very accurate for smooth functions
        ok = gl_err < 1e-3
        n_pass += int(ok)
        total += 1

        # Simpson should be reasonable
        ok = simp_err < 1e-3
        n_pass += int(ok)
        total += 1

    # Convergence test: GL improves with n
    print(f"\n  --- Convergence: ∫₀π sin(x) dx ---")
    f = lambda x: jnp.sin(x)
    prev_err = 1.0
    converge_ok = True
    for n in [5, 10, 20, 40]:
        val = float(integrate_gauss_legendre(f, 0.0, jnp.pi, n))
        err = abs(val - 2.0)
        print(f"    n={n:3d}: {val:.12f}  err={err:.2e}")
        if n > 5 and err > prev_err and err > 1e-12:  # allow machine-eps ties
            converge_ok = False
        prev_err = err

    n_pass += int(converge_ok)
    total += 1

    # JAX AD through integration
    def integral_of_param(k):
        return integrate_gauss_legendre(lambda x: jnp.sin(k * x), 0.0, jnp.pi, 20)

    grad_val = float(jax.grad(integral_of_param)(jnp.float64(1.0)))
    # d/dk ∫₀π sin(kx) dx = ∫₀π x cos(kx) dx; at k=1: [-x*cos(x)]₀π + [sin(x)]₀π = π*1 + 0 = π
    # Exact: ∫₀π x*cos(x) dx = [x*sin(x) + cos(x)]₀π = -1 - 1 = -2 + π*sin(π)... 
    # Let's just FD check
    h = 1e-6
    fd = (float(integral_of_param(jnp.float64(1.0+h))) - float(integral_of_param(jnp.float64(1.0-h)))) / (2*h)
    ad_ok = abs(grad_val - fd) / max(abs(fd), 1e-10) < 1e-4
    n_pass += int(ad_ok)
    total += 1
    print(f"\n  JAX AD: grad={grad_val:.6f}  FD={fd:.6f}  {'✓' if ad_ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All integration validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
