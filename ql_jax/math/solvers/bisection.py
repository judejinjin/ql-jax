"""Bisection root-finding method."""

_MAX_EVALUATIONS = 100


def solve(f, accuracy, x_min, x_max):
    """Find a root of f using bisection.

    Args:
        f: Scalar function f(x) -> float.
        accuracy: Target absolute accuracy.
        x_min, x_max: Bracket containing the root.

    Returns:
        The root x* such that |f(x*)| < accuracy.
    """
    fl = f(x_min)
    fh = f(x_max)

    if fl * fh > 0:
        raise ValueError("Bisection: root not bracketed")

    # Orient so f(xl) < 0
    if fl < 0:
        xl, xh = x_min, x_max
    else:
        xl, xh = x_max, x_min

    for _ in range(_MAX_EVALUATIONS):
        mid = 0.5 * (xl + xh)
        fmid = f(mid)

        if fmid < 0:
            xl = mid
        else:
            xh = mid

        if abs(xh - xl) < accuracy or fmid == 0:
            return mid

    raise RuntimeError("Bisection: max evaluations exceeded")
