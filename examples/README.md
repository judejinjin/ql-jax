# QL-JAX Examples

Runnable examples demonstrating QL-JAX capabilities.

| Example | Description |
|---------|-------------|
| `equity_option.py` | European option pricing with BSM, Heston, binomial, FD; AD Greeks; vmap batch pricing |
| `bonds.py` | Fixed-rate bond pricing with flat yield curve; DV01 via AD |
| `cds.py` | CDS pricing with midpoint engine; fair spread; AD hazard rate sensitivity |

## Running

```bash
cd ql-jax
source .venv/bin/activate
python examples/equity_option.py
python examples/bonds.py
python examples/cds.py
```
