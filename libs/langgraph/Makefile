.PHONY: all format lint test test_watch integration_tests spell_check spell_fix benchmark profile start-dev-server integration_tests

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING AND COVERAGE
######################

# Benchmarks

OUTPUT ?= out/benchmark.json

install:  ## Install dependencies
	uv sync --frozen --all-extras --all-packages --group dev

benchmark:
	mkdir -p out
	rm -f $(OUTPUT)
	uv run python -m bench -o $(OUTPUT) --rigorous

benchmark-fast:
	mkdir -p out
	rm -f $(OUTPUT)
	uv run python -m bench -o $(OUTPUT) --fast

GRAPH ?= bench/fanout_to_subgraph.py

profile:
	mkdir -p out
	sudo uv run py-spy record -g -o out/profile.svg -- python $(GRAPH)

# Run unit tests and generate a coverage report.
coverage:
	uv run pytest --cov \
		--cov-config=.coveragerc \
		--cov-report xml \
		--cov-report term-missing:skip-covered

start-services:
	docker compose -f tests/compose-postgres.yml -f tests/compose-redis.yml up -V --force-recreate --wait --remove-orphans

stop-services:
	docker compose -f tests/compose-postgres.yml -f tests/compose-redis.yml down -v

start-dev-server:
	LOG_LEVEL=warning uv run langgraph dev --config tests/example_app/langgraph.json --no-browser & echo "$$!" > .devserver.pid
	@echo "Dev server started."

stop-dev-server:
	@if [ -f .devserver.pid ]; then \
		kill `cat .devserver.pid` && rm .devserver.pid; \
		 echo "Dev server stopped."; \
	else \
		echo "No dev server PID file found."; \
	fi

TEST ?= .
NO_DOCKER ?= $(sh command -v docker >/dev/null 2>&1 && echo "false" || echo "true")

test:
	if [ "$(NO_DOCKER)" = "false" ]; then \
		make start-services &&\
		make start-dev-server &&\
		uv run pytest $(TEST); \
		EXIT_CODE=$$?; \
		make stop-services; \
		make stop-dev-server; \
		exit $$EXIT_CODE; \
	else \
		NO_DOCKER=true uv run pytest $(TEST) ; \
		EXIT_CODE=$$?; \
		exit $$EXIT_CODE; \
	fi

test_parallel:
	make start-services &&\
	make start-dev-server &&\
	uv run pytest -n auto --dist worksteal $(TEST) -vv --lf; \
	EXIT_CODE=$$?; \
	make stop-services; \
	make stop-dev-server; \
	exit $$EXIT_CODE

integration_tests:
	uv run pytest integration_tests

WORKERS ?= auto
XDIST_ARGS := $(if $(WORKERS),-n $(WORKERS) --dist worksteal,)
MAXFAIL ?=
MAXFAIL_ARGS := $(if $(MAXFAIL),--maxfail $(MAXFAIL),)
# Add an '-x' if xdist is enabled
XDIST_ARGS := $(if $(WORKERS),-x $(XDIST_ARGS),)

test_watch:
	make start-services &&\
	make start-dev-server &&\
	uv run ptw . -- --ff -vv $(XDIST_ARGS) $(MAXFAIL_ARGS) $(TEST); \
	EXIT_CODE=$$?; \
	make stop-services; \
	make stop-dev-server; \
	exit $$EXIT_CODE

test_watch_all:
	npx concurrently -n langgraph,checkpoint,checkpoint-sqlite,postgres "make test_watch" "make -C ../checkpoint test_watch" "make -C ../checkpoint-sqlite test_watch" "make -C ../checkpoint-postgres test_watch"


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E r'\.py$$|\.ipynb$$')
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
	uv run ruff check --select I --fix $(PYTHON_FILES)

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
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
