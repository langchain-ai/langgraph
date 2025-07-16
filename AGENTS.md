<general_rules>
This repository is a monorepo where each library lives in a subdirectory under `libs/`. When modifying code in any library, you must run the following commands in that library's directory before creating a pull request:

- `make format` – run code formatters (ruff)
- `make lint` – run the linter (ruff + mypy)
- `make test` – execute the test suite (pytest)

All libraries use `uv` as the package manager, not pip or poetry. Commands should be prefixed with `uv run` when executing tools like pytest, ruff, or mypy.

When creating new functionality, always search the relevant library's directory structure first to see if similar functionality exists. For the main `langgraph` library, check these key directories:
- `libs/langgraph/langgraph/channels/` - for channel-related functionality
- `libs/langgraph/langgraph/graph/` - for graph construction and state management
- `libs/langgraph/langgraph/pregel/` - for execution engine components
- `libs/langgraph/langgraph/utils/` - for utility functions

The root `Makefile` provides commands to run operations across all libraries:
- `make install` - install all library dependencies using uv
- `make format` - format code in all libraries
- `make lint` - lint all libraries
- `make test` - run tests in all libraries

Follow the existing code style enforced by ruff and ensure type annotations are present for mypy validation. All libraries maintain consistent configuration for these tools in their `pyproject.toml` files.
</general_rules>

<repository_structure>
This is a monorepo containing 8 main Python libraries under the `libs/` directory:

**Core Libraries:**
- `checkpoint` - base interfaces for LangGraph checkpointers
- `langgraph` - core framework for building stateful, multi-actor agents
- `prebuilt` - high-level APIs for creating and running agents and tools

**Storage Implementations:**
- `checkpoint-postgres` - PostgreSQL implementation of checkpoint saver
- `checkpoint-sqlite` - SQLite implementation of checkpoint saver

**Developer Tools:**
- `cli` - official command-line interface for LangGraph
- `sdk-py` - Python SDK for the LangGraph Platform API
- `sdk-js` - JavaScript/TypeScript SDK for LangGraph REST API

**Dependency Hierarchy:**
```
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
```

Each library follows a consistent structure with its own `pyproject.toml`, `Makefile`, `tests/` directory, and main package directory. The main `langgraph` library contains subdirectories for `channels/`, `func/`, `graph/`, `managed/`, `pregel/`, and `utils/`.

Libraries use editable installs to reference each other during development via `[tool.uv.sources]` sections in their `pyproject.toml` files.
</repository_structure>

<dependencies_and_installation>
This repository uses `uv` as the primary package manager. To set up the entire monorepo:

```bash
make install
```

This command creates a virtual environment and installs all libraries in editable mode using `uv pip install -e` for each library in `libs/`.

Each library has its own `pyproject.toml` with:
- Production dependencies in the `[project]` dependencies array
- Development dependencies in `[dependency-groups]` dev array
- Local library references in `[tool.uv.sources]` for editable installs

Common development dependencies across all libraries include: `ruff`, `codespell`, `pytest`, `pytest-asyncio`, `pytest-mock`, `mypy`, and `pytest-watcher`.

To work on a specific library, navigate to its directory under `libs/` and use `uv run` to execute commands within that library's environment.
</dependencies_and_installation>

<testing_instructions>
All libraries use `pytest` as the testing framework with asyncio support enabled (`asyncio_mode = "auto"`). Tests are configured with strict markers and detailed output (`--strict-markers --strict-config --durations=5 -vv`).

**Running Tests:**
- Single library: `cd libs/<library> && make test`
- All libraries: `make test` (from root)
- Specific test file: `make test TEST=path/to/test.py`
- Watch mode: `make test_watch` (in individual library directories)

**PostgreSQL-dependent Libraries:**
Some libraries (`langgraph`, `checkpoint-postgres`, `prebuilt`) require PostgreSQL for testing. These use Docker Compose with configuration files at `libs/*/tests/compose-postgres.yml`. The test commands automatically start/stop PostgreSQL containers.

To skip Docker-dependent tests, set `NO_DOCKER=true` environment variable.

**Test Structure:**
- Test files are located in `libs/*/tests/` directories
- Use `pytest-asyncio` for async test functions
- Common test utilities and fixtures are shared via `conftest.py` files
- Some libraries use `syrupy` for snapshot testing

Tests should cover core functionality, error handling, and integration scenarios. Follow existing test patterns within each library for consistency.
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

