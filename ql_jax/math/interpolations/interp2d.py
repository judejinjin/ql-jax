"""2D interpolation methods — bilinear and bicubic."""

import jax.numpy as jnp


def bilinear_build(xs, ys, zs):
    """Build bilinear interpolation on a rectangular grid.

    Parameters
    ----------
    xs : 1D array of x grid points (sorted)
    ys : 1D array of y grid points (sorted)
    zs : 2D array of shape (len(xs), len(ys))
    """
    return {
        "xs": jnp.asarray(xs, dtype=jnp.float64),
        "ys": jnp.asarray(ys, dtype=jnp.float64),
        "zs": jnp.asarray(zs, dtype=jnp.float64),
    }


def bilinear_evaluate(state, x, y):
    """Evaluate bilinear interpolation at (x, y)."""
    xs, ys, zs = state["xs"], state["ys"], state["zs"]

    ix = jnp.searchsorted(xs, x, side="right") - 1
    ix = jnp.clip(ix, 0, len(xs) - 2)
    iy = jnp.searchsorted(ys, y, side="right") - 1
    iy = jnp.clip(iy, 0, len(ys) - 2)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]

    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)

    z00 = zs[ix, iy]
    z10 = zs[ix + 1, iy]
    z01 = zs[ix, iy + 1]
    z11 = zs[ix + 1, iy + 1]

    return (z00 * (1 - tx) * (1 - ty)
            + z10 * tx * (1 - ty)
            + z01 * (1 - tx) * ty
            + z11 * tx * ty)


def bicubic_build(xs, ys, zs):
    """Build bicubic spline interpolation on a rectangular grid.

    Uses natural boundary conditions.
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    zs = jnp.asarray(zs, dtype=jnp.float64)

    return {"xs": xs, "ys": ys, "zs": zs}


def bicubic_evaluate(state, x, y):
    """Evaluate bicubic interpolation at (x, y).

    Uses Catmull-Rom cubic interpolation kernel.
    """
    xs, ys, zs = state["xs"], state["ys"], state["zs"]
    nx, ny = zs.shape

    ix = jnp.searchsorted(xs, x, side="right") - 1
    ix = jnp.clip(ix, 1, nx - 3)
    iy = jnp.searchsorted(ys, y, side="right") - 1
    iy = jnp.clip(iy, 1, ny - 3)

    tx = (x - xs[ix]) / (xs[ix + 1] - xs[ix])
    ty = (y - ys[iy]) / (ys[iy + 1] - ys[iy])

    def cubic_kernel(t):
        """Catmull-Rom basis values at t in [0,1]."""
        return jnp.array([
            -0.5 * t**3 + t**2 - 0.5 * t,
            1.5 * t**3 - 2.5 * t**2 + 1.0,
            -1.5 * t**3 + 2.0 * t**2 + 0.5 * t,
            0.5 * t**3 - 0.5 * t**2,
        ])

    wx = cubic_kernel(tx)
    wy = cubic_kernel(ty)

    result = 0.0
    for di in range(4):
        for dj in range(4):
            result += wx[di] * wy[dj] * zs[ix - 1 + di, iy - 1 + dj]

    return result
