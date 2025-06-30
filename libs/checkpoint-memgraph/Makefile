# libs/checkpoint-memgraph/Makefile
.PHONY: test test_watch lint format start-memgraph stop-memgraph

######################
# TESTING AND COVERAGE
######################

start-memgraph:
	MEMGRAPH_VERSION=${MEMGRAPH_VERSION:-3.3} docker compose -f tests/compose-memgraph.yml up -V --force-recreate --wait || ( \
		echo "Failed to start Memgraph, printing logs..."; \
		docker compose -f tests/compose-memgraph.yml logs; \
		exit 1 \
	)

stop-memgraph:
	docker compose -f tests/compose-memgraph.yml down

MEMGRAPH_VERSIONS ?= 3.3

test_memgraph_version:
	@echo "Testing Memgraph $(MEMGRAPH_VERSION)"
	@MEMGRAPH_VERSION=$(MEMGRAPH_VERSION) make start-memgraph
	@uv run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-memgraph; \
	echo "Finished testing Memgraph $(MEMGRAPH_VERSION); Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE

test:
	@for version in $(MEMGRAPH_VERSIONS); do \
		if ! make test_memgraph_version MEMGRAPH_VERSION=$$version; then \
			echo "Test failed for Memgraph $$version"; \
			exit 1; \
		fi; \
	done
	@echo "All Memgraph versions tested successfully"

TEST ?= .
test_watch:
	MEMGRAPH_VERSION=${MEMGRAPH_VERSION:-2.13} make start-memgraph; \
	uv run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-memgraph; \
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
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)
