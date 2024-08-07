.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-postgres:
	docker compose -f tests/compose-postgres.yml up -V --force-recreate --wait

stop-postgres:
	docker compose -f tests/compose-postgres.yml down

test:
	make start-postgres; \
	poetry run pytest; \
	EXIT_CODE=$$?; \
	make stop-postgres; \
	exit $$EXIT_CODE

test_watch:
	make start-postgres; \
	poetry run ptw .; \
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
	poetry run ruff .
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) || poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff --select I --fix $(PYTHON_FILES)
