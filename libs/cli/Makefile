.PHONY: test lint format test-integration update-schema build build-release clean cargo-test cargo-lint

######################
# RUST BUILD
######################

build:
	cargo build

build-release:
	cargo build --release

cargo-test:
	cargo test

cargo-lint:
	cargo fmt --check
	cargo clippy -- -D warnings

cargo-fmt:
	cargo fmt

######################
# PYTHON TESTING AND COVERAGE
######################

TEST?= "tests/unit_tests"
test:
	uv run pytest $(TEST)
test-integration:
	uv run pytest tests/integration_tests

######################
# PYTHON LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph_cli
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

######################
# PACKAGING
######################

# Build maturin wheel for current platform (development)
wheel-dev:
	maturin build --release

# Build maturin wheel for specific target
# Usage: make wheel TARGET=x86_64-unknown-linux-gnu
TARGET?=
wheel:
	maturin build --release $(if $(TARGET),--target $(TARGET),)

# Build sdist
sdist:
	maturin sdist

######################
# SCHEMA
######################

update-schema:
	uv run python generate_schema.py

######################
# CLEANUP
######################

clean:
	cargo clean
	rm -rf dist/ target/wheels/
