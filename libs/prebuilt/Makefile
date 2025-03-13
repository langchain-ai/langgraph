.PHONY: all format lint test test_watch integration_tests spell_check spell_fix benchmark profile

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

start-postgres:
	docker compose -f tests/compose-postgres.yml up -V --force-recreate --wait --remove-orphans

stop-postgres:
	docker compose -f tests/compose-postgres.yml down -v

TEST ?= .

test:
	make start-postgres && poetry run pytest $(TEST); \
	EXIT_CODE=$$?; \
	make stop-postgres; \
	exit $$EXIT_CODE

test_watch:
	make start-postgres && poetry run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-postgres; \
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
	poetry run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy langgraph --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w


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
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
