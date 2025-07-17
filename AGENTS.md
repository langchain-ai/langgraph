<general_rules>
This repository is a monorepo where each library lives in a subdirectory under `libs/`. When modifying code in any library, you must run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters (ruff)
- `make lint` – run linters (ruff and mypy)
- `make test` – execute the test suite

To run a specific test file or pass additional pytest options, use the `TEST` variable:
```bash
TEST=path/to/test.py make test
TEST="tests/test_specific.py -v" make test
```

All libraries use consistent code quality tools:
- **ruff** for code formatting and linting (replaces black and flake8)
- **mypy** for static type checking
- All commands are prefixed with `uv run` in library Makefiles

When creating new functionality, always search existing code in the relevant library first to avoid duplication. Follow the existing patterns for imports, error handling, and async/sync implementations found in each library.

Root-level commands are available to operate across all libraries:
- `make install` – install all library dependencies
- `make format` – format code in all libraries
- `make lint` – lint all libraries
- `make test` – run tests in all libraries
</general_rules>

<repository_structure>
This is a monorepo containing 8 main libraries in the `libs/` directory:

**Core Libraries:**
- **langgraph** – Core framework for building stateful, multi-actor agents
- **checkpoint** – Base interfaces for LangGraph checkpointers
- **prebuilt** – High-level APIs for creating and running agents and tools

**Checkpoint Implementations:**
- **checkpoint-postgres** – PostgreSQL implementation of checkpoint saver
- **checkpoint-sqlite** – SQLite implementation of checkpoint saver

**Platform Integration:**
- **cli** – Official command-line interface for LangGraph
- **sdk-py** – Python SDK for LangGraph Platform API
- **sdk-js** – JavaScript/TypeScript SDK for LangGraph REST API

**Dependency Relationships:**
```
checkpoint → checkpoint-postgres, checkpoint-sqlite, prebuilt, langgraph
prebuilt → langgraph
sdk-py → langgraph, cli
sdk-js (standalone)
```

Each library contains:
- `pyproject.toml` – dependency and tool configuration
- `Makefile` – standardized build targets
- `tests/` – test files with fixtures and conftest.py
- `uv.lock` – locked dependency versions

The `docs/` directory contains documentation built with MkDocs, and `examples/` contains Jupyter notebooks demonstrating LangGraph features.
</repository_structure>

<dependencies_and_installation>
This repository uses **uv** as the primary package manager (not pip or poetry). Dependencies are managed through `pyproject.toml` files with `uv.lock` files for version locking.

**Installation:**
```bash
make install  # Installs all libraries with dependencies
```

This command creates a virtual environment and installs all libraries in `libs/` as editable packages using `uv pip install -e`.

**Library Interdependencies:**
Libraries reference each other through `uv.sources` sections in their `pyproject.toml` files using path-based editable installs. For example:
```toml
[tool.uv.sources]
langgraph-checkpoint = { path = "../checkpoint", editable = true }
```

**Development Dependencies:**
Each library defines dev dependencies in `dependency-groups.dev` sections, typically including pytest, ruff, mypy, and library-specific testing tools.
</dependencies_and_installation>

<testing_instructions>
The repository uses **pytest** as the testing framework with consistent configuration across all libraries:

**Running Tests:**
```bash
make test                    # Run all tests in current library
TEST=path/to/test.py make test  # Run specific test file
```

**Key Testing Features:**
- **asyncio_mode = "auto"** configured in all libraries for seamless async testing
- **pytest-asyncio**, **pytest-mock**, **pytest-watcher** plugins available
- **syrupy** for snapshot testing
- **pytest-cov** for coverage reporting

**Integration Testing:**
Some libraries require **Docker Compose** for PostgreSQL integration tests:
- PostgreSQL services defined in `tests/compose-postgres.yml` files
- Tests automatically start/stop Docker services via Makefile targets
- Use `NO_DOCKER=true` environment variable to skip Docker-dependent tests

**Test Structure:**
- Each library has `tests/` directory with `conftest.py` for fixtures
- Test files follow `test_*.py` naming convention
- Libraries include both unit tests and integration tests
- Some libraries have benchmarking capabilities in `bench/` directories

**Coverage:**
Run `make coverage` in individual libraries to generate coverage reports with pytest-cov.
</testing_instructions>
# AGENTS Instructions

This repository is a monorepo. Each library lives in a subdirectory under `libs/`.

When you modify code in any library, run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters
- `make lint` – run the linter
- `make test` – execute the test suite

To run a particular test file or to pass additional pytest options you can specify the `TEST` variable:

```
TEST=path/to/test.py make test
```

Other pytest arguments can also be supplied inside the `TEST` variable.

## Libraries

The repository contains several Python and JavaScript/TypeScript libraries.
Below is a high-level overview:

- **checkpoint** – base interfaces for LangGraph checkpointers.
- **checkpoint-postgres** – Postgres implementation of the checkpoint saver.
- **checkpoint-sqlite** – SQLite implementation of the checkpoint saver.
- **cli** – official command-line interface for LangGraph.
- **langgraph** – core framework for building stateful, multi-actor agents.
- **prebuilt** – high-level APIs for creating and running agents and tools.
- **sdk-js** – JS/TS SDK for interacting with the LangGraph REST API.
- **sdk-py** – Python SDK for the LangGraph Platform API.

### Dependency map

The diagram below lists downstream libraries for each production dependency as
declared in that library's `pyproject.toml` (or `package.json`).

```text
checkpoint
├── checkpoint-postgres
├── checkpoint-sqlite
├── prebuilt
└── langgraph

prebuilt
└── langgraph

sdk-py
├── langgraph
└── cli

sdk-js (standalone)
```

Changes to a library may impact all of its dependents shown above.

