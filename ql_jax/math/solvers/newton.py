"""Newton's root-finding method.

Uses the derivative of f for quadratic convergence.
Falls back to bisection if Newton step goes out of bracket.
"""


_MAX_EVALUATIONS = 100


def solve(f, df, accuracy, guess, x_min=None, x_max=None):
    """Find a root of f using Newton's method with derivative df.

    Args:
        f: Scalar function f(x) -> float.
        df: Derivative function df(x) -> float.
        accuracy: Target absolute accuracy.
        guess: Initial guess.
        x_min, x_max: Optional bracket for safety.

    Returns:
        The root x* such that |f(x*)| < accuracy.
    """
    x = guess

    if x_min is not None and x_max is not None:
        # Newton-safe: bracket-confined
        return _newton_safe(f, df, accuracy, guess, x_min, x_max)

    # Pure Newton
    for _ in range(_MAX_EVALUATIONS):
        fx = f(x)
        if abs(fx) < accuracy:
            return x
        dfx = df(x)
        if dfx == 0:
            raise RuntimeError("Newton: zero derivative")
        x -= fx / dfx

    raise RuntimeError("Newton: max evaluations exceeded")


def _newton_safe(f, df, accuracy, guess, x_min, x_max):
    """Newton's method with bisection fallback."""
    fl = f(x_min)
    fh = f(x_max)

    if fl * fh > 0:
        raise ValueError("Newton-safe: root not bracketed")

    if fl == 0:
        return x_min
    if fh == 0:
        return x_max

    # Orient so f(xl) < 0
    if fl < 0:
        xl, xh = x_min, x_max
    else:
        xl, xh = x_max, x_min

    root = guess
    dx_old = abs(x_max - x_min)
    dx = dx_old

    froot = f(root)
    dfroot = df(root)

    for _ in range(_MAX_EVALUATIONS):
        # Check if Newton step is out of range or not decreasing fast enough
        if (((root - xh) * dfroot - froot) * ((root - xl) * dfroot - froot) > 0
                or abs(2.0 * froot) > abs(dx_old * dfroot)):
            # Bisection
            dx_old = dx
            dx = 0.5 * (xh - xl)
            root = xl + dx
        else:
            dx_old = dx
            dx = froot / dfroot
            root -= dx

        if abs(dx) < accuracy:
            return root

        froot = f(root)
        dfroot = df(root)

        if froot < 0:
            xl = root
        else:
            xh = root

    raise RuntimeError("Newton-safe: max evaluations exceeded")
