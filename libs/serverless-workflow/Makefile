.PHONY: all format lint type test test_watch help

all: help

######################
# TESTING AND COVERAGE
######################

TEST ?= .

test:
	uv run pytest $(TEST)

test_watch:
	uv run ptw $(TEST)

######################
# LINTING AND FORMATTING
######################
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph_serverless_workflow
lint_tests: PYTHON_FILES=tests

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run ty check langgraph_serverless_workflow

type:
	uv run ty check langgraph_serverless_workflow

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --fix $(PYTHON_FILES)

######################
# HELP
######################
help:
	@echo '-- LINTING --'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo '-- TESTS --'
	@echo 'test                         - run unit tests'
	@echo 'test TEST=<test_file>       - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
