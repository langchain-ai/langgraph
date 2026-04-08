.PHONY: test lint type format test-integration update-schema bump-version
.PHONY: test-go lint-go format-go

######################
# TESTING AND COVERAGE
######################

TEST?= "tests/unit_tests"
GO_FILES=$(shell find cmd internal -type f -name '*.go' 2>/dev/null)
test: test-go
	uv run pytest $(TEST)
test-integration:
	uv run pytest tests/integration_tests

test-go:
	[ ! -f go.mod ] || go test ./...

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph_cli
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests: lint-go
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint-go:
	[ -z "$(GO_FILES)" ] || test -z "$$(gofmt -l $(GO_FILES))"

type:
	mkdir -p $(MYPY_CACHE) && uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff: format-go
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

format-go:
	[ -z "$(GO_FILES)" ] || gofmt -w $(GO_FILES)

update-schema:
	uv run python generate_schema.py

bump-version:
	uv run hatch version patch
