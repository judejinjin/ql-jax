"""Shared utilities for ql-jax demo notebooks."""

import os
import sys
import time
import statistics
import numpy as np

# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def setup_backend(backend="cpu"):
    """Configure JAX backend and enable float64.

    Call this at the top of every notebook BEFORE importing jax or ql_jax.

    Parameters
    ----------
    backend : str
        ``"cpu"`` or ``"gpu"``.
    """
    os.environ["JAX_PLATFORMS"] = backend
    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX backend : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}")
    return jax

# ---------------------------------------------------------------------------
# QuantLib SWIG loader
# ---------------------------------------------------------------------------

_QL = None

def load_quantlib():
    """Import QuantLib-SWIG 1.42-rc with correct paths.

    Returns the ``QuantLib`` module.
    """
    global _QL
    if _QL is not None:
        return _QL
    ql_lib = "/home/jude/QuantLib/build/ql"
    ql_py  = "/home/jude/QuantLib-SWIG/Python/build/lib.linux-x86_64-cpython-312"
    os.environ["LD_LIBRARY_PATH"] = ql_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    if ql_py not in sys.path:
        sys.path.insert(0, ql_py)
    import QuantLib
    _QL = QuantLib
    print(f"QuantLib    : {QuantLib.__version__}")
    return QuantLib

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare(label, ql_val, jax_val, unit="", tol=1e-6):
    """Print a single comparison row.

    Returns ``True`` if values agree within *tol* (absolute) or 0.01 %
    (relative).
    """
    if isinstance(ql_val, (int, float, np.floating)):
        abs_err = abs(float(ql_val) - float(jax_val))
        denom = max(abs(float(ql_val)), 1e-15)
        rel_err = abs_err / denom
        ok = abs_err < tol or rel_err < 1e-4
        status = "✅" if ok else "❌"
        print(f"  {status} {label:40s}  QL={float(ql_val):>18.10f}  "
              f"JAX={float(jax_val):>18.10f}  "
              f"err={abs_err:.2e}  rel={rel_err:.2e}  {unit}")
        return ok
    # Non-numeric fallback
    ok = ql_val == jax_val
    status = "✅" if ok else "❌"
    print(f"  {status} {label:40s}  QL={ql_val}  JAX={jax_val}  {unit}")
    return ok


def compare_table(rows, title="Comparison"):
    """Print a markdown-friendly comparison table.

    *rows* is a list of ``(label, ql_val, jax_val)`` or
    ``(label, ql_val, jax_val, unit)`` tuples.

    Returns the number of failures.
    """
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    fails = 0
    for row in rows:
        unit = row[3] if len(row) > 3 else ""
        if not compare(row[0], row[1], row[2], unit):
            fails += 1
    tag = "ALL PASS ✅" if fails == 0 else f"{fails} FAILURE(S) ❌"
    print(f"{'─'*80}")
    print(f"  {tag}  ({len(rows)} checks)")
    print(f"{'='*80}\n")
    return fails

# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def timed(fn, *args, warmup=2, runs=5, **kwargs):
    """Benchmark *fn* and return ``(avg_seconds, result)``.

    The function is called *warmup* + *runs* times.  Only the last *runs*
    are measured.
    """
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        # Block on JAX async dispatch
        try:
            import jax
            result_flat = jax.tree.leaves(result)
            for leaf in result_flat:
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    avg = statistics.mean(times)
    return avg, result


def timed_ms(fn, *args, warmup=2, runs=5, **kwargs):
    """Like :func:`timed` but returns ``(avg_milliseconds, result)``."""
    avg, result = timed(fn, *args, warmup=warmup, runs=runs, **kwargs)
    return avg * 1000, result

# ---------------------------------------------------------------------------
# Performance visualization
# ---------------------------------------------------------------------------

def plot_speedup(labels, ql_times, jax_times, title="QuantLib vs ql-jax"):
    """Bar chart comparing QuantLib and ql-jax timings.

    Parameters
    ----------
    labels : list[str]
        Operation names.
    ql_times, jax_times : list[float]
        Timings in the same unit (seconds or ms).
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, ql_times, width, label="QuantLib", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, jax_times, width, label="ql-jax (JIT)", color="#ff7f0e")

    ax.set_ylabel("Time")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    # Annotate speedup
    for i, (qt, jt) in enumerate(zip(ql_times, jax_times)):
        if jt > 0:
            speedup = qt / jt
            ax.annotate(f"{speedup:.1f}×",
                        xy=(x[i] + width / 2, jt),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, color="#ff7f0e")

    fig.tight_layout()
    plt.show()


def plot_convergence(ns, values, exact=None, xlabel="N", ylabel="Price",
                     title="Convergence"):
    """Plot convergence of a numerical method."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, values, "o-", label="Computed")
    if exact is not None:
        ax.axhline(exact, color="r", linestyle="--", label=f"Exact = {exact:.6f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_heatmap(matrix, xlabels, ylabels, title="Heatmap", fmt=".4f",
                 cmap="RdBu_r"):
    """Plot a matrix as a labeled heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    # Annotate cells
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(matrix[i, j]) > 0.5 * np.max(np.abs(matrix)) else "black")

    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# ql-jax path helper
# ---------------------------------------------------------------------------

# Ensure ql-jax root is on sys.path
_ql_jax_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ql_jax_root not in sys.path:
    sys.path.insert(0, _ql_jax_root)
