"""Secant root-finding method."""

_MAX_EVALUATIONS = 100


def solve(f, accuracy, guess, step=1.0):
    """Find a root of f using the secant method.

    Args:
        f: Scalar function f(x) -> float.
        accuracy: Target absolute accuracy.
        guess: Initial guess.
        step: Step for second point.

    Returns:
        The root x*.
    """
    x0 = guess
    x1 = guess + step
    f0 = f(x0)
    f1 = f(x1)

    for _ in range(_MAX_EVALUATIONS):
        if abs(f1) < accuracy:
            return x1
        if f1 == f0:
            raise RuntimeError("Secant: division by zero (f values equal)")
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0 = x1, f1
        x1 = x_new
        f1 = f(x1)

    raise RuntimeError("Secant: max evaluations exceeded")
