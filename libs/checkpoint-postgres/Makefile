.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-postgres:
	POSTGRES_VERSION=${POSTGRES_VERSION:-16} docker compose -f tests/compose-postgres.yml up -V --force-recreate --wait || ( \
		echo "Failed to start PostgreSQL, printing logs..."; \
		docker compose -f tests/compose-postgres.yml logs; \
		exit 1 \
	)

stop-postgres:
	docker compose -f tests/compose-postgres.yml down

POSTGRES_VERSIONS ?= 15 16
test_pg_version:
	@echo "Testing PostgreSQL $(POSTGRES_VERSION)"
	@POSTGRES_VERSION=$(POSTGRES_VERSION) make start-postgres
	@uv run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-postgres; \
	echo "Finished testing PostgreSQL $(POSTGRES_VERSION); Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE

test:
	@for version in $(POSTGRES_VERSIONS); do \
		if ! make test_pg_version POSTGRES_VERSION=$$version; then \
			echo "Test failed for PostgreSQL $$version"; \
			exit 1; \
		fi; \
	done
	@echo "All PostgreSQL versions tested successfully"

TEST ?= .
test_watch:
	POSTGRES_VERSION=${POSTGRES_VERSION:-16} make start-postgres; \
	uv run ptw $(TEST); \
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
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)
