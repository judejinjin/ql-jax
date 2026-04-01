.PHONY: venv install install-dev test bench docs lint clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

bench:
	$(PYTHON) -m pytest benchmarks/ --benchmark-only

docs:
	$(VENV)/bin/sphinx-build -b html docs/ docs/_build/html

lint:
	$(VENV)/bin/ruff check ql_jax/ tests/
	$(VENV)/bin/black --check ql_jax/ tests/

format:
	$(VENV)/bin/black ql_jax/ tests/
	$(VENV)/bin/ruff check --fix ql_jax/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
