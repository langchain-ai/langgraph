.PHONY: test lint type format test-integration update-schema bump-version
.PHONY: test-go lint-go format-go
.PHONY: build-go build-go-all clean-go-bin

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

######################
# GO BINARY CROSS-COMPILATION
######################

GO_MODULE=github.com/langchain-ai/langgraph/libs/cli
GO_BINARY=cmd/langgraph/main.go
GO_VERSION_PKG=$(GO_MODULE)/internal/version
GO_BIN_DIR=langgraph_cli/bin
GO_VERSION=$(shell grep -m 1 '^__version__' langgraph_cli/__init__.py | cut -d '"' -f 2)
GO_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GO_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GO_LDFLAGS=-s -w \
  -X '$(GO_VERSION_PKG).Version=$(GO_VERSION)' \
  -X '$(GO_VERSION_PKG).Commit=$(GO_COMMIT)' \
  -X '$(GO_VERSION_PKG).Date=$(GO_DATE)'

# Platform matrix — covers every platform orjson ships for. Since the Go
# binary is statically linked (CGO_ENABLED=0), the same linux binary works
# on both glibc and musl. Musllinux wheels reuse the linux binary but are
# tagged differently so pip installs them on Alpine/musl systems.
GO_PLATFORMS = \
  linux/amd64 \
  linux/arm64 \
  linux/arm \
  linux/386 \
  linux/ppc64le \
  linux/s390x \
  darwin/amd64 \
  darwin/arm64 \
  windows/amd64 \
  windows/arm64 \
  windows/386

# Build for the current platform (development).
build-go:
	@mkdir -p $(GO_BIN_DIR)
	go build -ldflags "$(GO_LDFLAGS)" -o $(GO_BIN_DIR)/langgraph $(GO_BINARY)
	@echo "Built $(GO_BIN_DIR)/langgraph"

# Build for a single target: make build-go-target GOOS=linux GOARCH=amd64
build-go-target:
	$(eval EXT=$(if $(filter windows,$(GOOS)),.exe,))
	@mkdir -p $(GO_BIN_DIR)
	GOOS=$(GOOS) GOARCH=$(GOARCH) CGO_ENABLED=0 \
		go build -ldflags "$(GO_LDFLAGS)" \
		-o $(GO_BIN_DIR)/langgraph-$(GOOS)-$(GOARCH)$(EXT) $(GO_BINARY)
	@echo "Built $(GO_BIN_DIR)/langgraph-$(GOOS)-$(GOARCH)$(EXT)"

# Build for all platforms.
build-go-all:
	@mkdir -p $(GO_BIN_DIR)
	@for platform in $(GO_PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		ext=""; \
		if [ "$$os" = "windows" ]; then ext=".exe"; fi; \
		echo "Building $$os/$$arch..."; \
		GOOS=$$os GOARCH=$$arch CGO_ENABLED=0 \
			go build -ldflags "$(GO_LDFLAGS)" \
			-o $(GO_BIN_DIR)/langgraph-$$os-$$arch$$ext $(GO_BINARY) || exit 1; \
	done
	@echo "All platforms built in $(GO_BIN_DIR)/"

clean-go-bin:
	rm -rf $(GO_BIN_DIR)

######################
# SCHEMA AND VERSIONING
######################

update-schema:
	uv run python generate_schema.py

bump-version:
	uv run hatch version patch
