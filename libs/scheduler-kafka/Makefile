.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-services:
	docker compose -f tests/compose.yml up -V --force-recreate --wait --remove-orphans

stop-services:
	docker compose -f tests/compose.yml down

TEST_PATH ?= .

test:
	make start-services && uv run pytest $(TEST_PATH); \
	EXIT_CODE=$$?; \
	make stop-services; \
	exit $$EXIT_CODE

test_watch:
	make start-services && uv run ptw . -- -x $(TEST_PATH); \
	EXIT_CODE=$$?; \
	make stop-services; \
	exit $$EXIT_CODE

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
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
