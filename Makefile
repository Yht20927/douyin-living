# Douyin Live AI Clipper — common developer commands.
# Run `make help` for the available targets.

PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python

.PHONY: help setup test cov lint e2e models check clean

help:
	@echo "Targets:"
	@echo "  make setup   Create .venv and install base + dev dependencies"
	@echo "  make test    Run the test suite"
	@echo "  make cov     Run tests with coverage (fail below 75%)"
	@echo "  make lint    Run ruff"
	@echo "  make e2e     Run the synthetic end-to-end smoke test"
	@echo "  make models  Download/check optional model files"
	@echo "  make check   Run environment pre-flight check"
	@echo "  make clean   Remove caches and build artifacts"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m coverage run -m pytest -q
	$(PY) -m coverage report -m --fail-under=75

lint:
	$(PY) -m ruff check src tests

e2e:
	$(PY) -m pytest tests/ -q -k "e2e or end_to_end"

models:
	$(PY) scripts/download_models.py --check

check:
	$(PY) check_env.py

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
