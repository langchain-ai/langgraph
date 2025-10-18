.PHONY: all format lint test test-fast test_watch integration_tests spell_check spell_fix benchmark profile

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

start-services:
	docker compose -f tests/compose-postgres.yml -f tests/compose-redis.yml up -V --force-recreate --wait --remove-orphans

stop-services:
	docker compose -f tests/compose-postgres.yml -f tests/compose-redis.yml down -v

TEST ?= .

test-fast:
	LANGGRAPH_TEST_FAST=1 uv run pytest $(TEST)

test:
	make start-services && LANGGRAPH_TEST_FAST=0 uv run --active pytest $(TEST); \
	EXIT_CODE=$$?; \
	make stop-services; \
	exit $$EXIT_CODE

test_watch:
	make start-services && LANGGRAPH_TEST_FAST=0 uv run ptw $(TEST); \
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
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy langgraph --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w


######################
# HELP
######################

help:
	@echo '===================='
	@echo '-- DOCUMENTATION --'
	
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'spell_check               	- run codespell on the project'
	@echo 'spell_fix               		- run codespell on the project and fix the errors'
	@echo '-- TESTS --'
	@echo 'coverage                     - run unit tests and generate coverage report'
	@echo 'test                         - run unit tests'
	@echo 'test-fast                    - run unit tests with in-memory checkpointer only'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
