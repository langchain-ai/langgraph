.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-sqlserver:
	docker compose -f tests/compose-sqlserver.yml up -V --force-recreate --wait || ( \
		echo "Failed to start SQL Server, printing logs..."; \
		docker compose -f tests/compose-sqlserver.yml logs; \
		exit 1 \
	)

stop-sqlserver:
	docker compose -f tests/compose-sqlserver.yml down

test:
	@echo "Testing SQLServer"
	@make start-sqlserver
	@uv run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-sqlserver; \
	if [ $$EXIT_CODE -ne 0 ]; then \
		echo "Test failed for SQLServer with exit code $$EXIT_CODE"; \
	else \
		echo "SQLServer tested successfully"; \
	fi; \
	exit $$EXIT_CODE

TEST ?= .
test_watch:
	make start-sqlserver; \
	uv run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-sqlserver; \
	exit $$EXIT_CODE

########################
# LINTING AND FORMATTING
########################

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
