PY ?= python

.PHONY: help install install-dev test test-fast lint format notebooks benchmark clean build

help:
	@echo "install       install the package with the vine extra"
	@echo "install-dev   install with all dev/test/notebook extras"
	@echo "test          run the full test suite"
	@echo "test-fast     run tests, skipping the slow model-fitting ones"
	@echo "lint          ruff check"
	@echo "format        ruff check --fix + ruff format"
	@echo "notebooks     execute every notebook in examples/ and benchmarks/ in place"
	@echo "benchmark     run the full benchmark suite to benchmarks/results/"
	@echo "build         build the sdist and wheel"

install:
	$(PY) -m pip install -e ".[vine]"

install-dev:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 examples/*.ipynb benchmarks/*.ipynb

benchmark:
	$(PY) benchmarks/run_benchmark.py

build:
	$(PY) -m pip install --upgrade build
	$(PY) -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
