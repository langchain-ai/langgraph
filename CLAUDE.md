# Claude Code Assistant Guide for LangGraph

## Project Overview

**LangGraph** is a low-level orchestration framework for building, managing, and deploying long-running, stateful AI agents and workflows. This is a **Python-based monorepo** with additional JavaScript/TypeScript SDK components.

### Key Characteristics
- **Type**: Multi-language monorepo (primarily Python with JS/TS SDKs)
- **Architecture**: Library + Platform + CLI + Documentation
- **Primary Language**: Python (3.9+) 
- **Build System**: `uv` (modern Python package manager) + `make`
- **Framework**: Based on Pregel algorithm (Google's distributed computing model)

## Repository Structure

### Core Libraries (libs/)
```
libs/
├── langgraph/           # Core LangGraph framework
├── checkpoint/          # Base checkpointing functionality 
├── checkpoint-sqlite/   # SQLite checkpoint implementation
├── checkpoint-postgres/ # PostgreSQL checkpoint implementation
├── prebuilt/           # Pre-built agent components
├── cli/                # LangGraph CLI tool
├── sdk-py/             # Python SDK for LangGraph Platform
└── sdk-js/             # JavaScript/TypeScript SDK
```

### Documentation (docs/)
- **Location**: `/docs/` directory
- **Build System**: MkDocs with Material theme
- **Content**: Mix of Markdown (.md) and Jupyter Notebooks (.ipynb)
- **Structure**: Follows Diataxis framework (Tutorials, How-tos, References, Explanations)

### Examples
- **Location**: `/examples/` directory
- **Note**: Used for testing/development only, NOT for documentation
- **Documentation**: All new docs go in `/docs/docs/`

## Development Workflow

### Prerequisites
- **Python**: 3.9+ required
- **Package Manager**: `uv` (recommended) or `pip`
- **Docker**: Optional, for testing with databases
- **Node.js**: For JavaScript/TypeScript components

### Setup & Installation
```bash
# Install all dependencies across all libraries
make install

# Or for individual libraries
cd libs/langgraph
uv sync --frozen --all-extras --all-packages --group dev
```

### Core Commands

#### Development Commands (from repo root)
```bash
make all           # Run full CI pipeline: lint, format, lock, test, codespell
make install       # Install dependencies for all projects
make lint          # Lint all projects
make format        # Format all projects  
make test          # Test all projects
make lock          # Update dependency locks
make codespell     # Spell check all projects
```

#### Individual Library Commands (from libs/[library]/)
```bash
make test                    # Run tests (starts postgres/dev-server if needed)
make test_parallel          # Run tests in parallel
make test_watch             # Run tests in watch mode
make lint                   # Lint code with ruff + mypy
make format                 # Format code with ruff
make coverage               # Generate coverage report
make benchmark              # Run performance benchmarks
```

### Testing Infrastructure
- **Framework**: pytest with extensive fixtures
- **Coverage**: pytest-cov
- **Docker**: Auto-starts PostgreSQL container for integration tests
- **Dev Server**: Auto-starts LangGraph dev server for testing
- **Parallel**: pytest-xdist for parallel execution

### Code Quality Tools
- **Linting**: ruff (replaces flake8, isort, etc.)
- **Type Checking**: mypy with strict configuration
- **Formatting**: ruff format (replaces black)
- **Spell Check**: codespell with pre-commit hooks
- **Git Hooks**: pre-commit for automatic checks

## Architecture Deep Dive

### Core Components

#### 1. Pregel Engine (`langgraph/pregel/`)
- **Purpose**: Core execution engine based on Google's Pregel algorithm
- **Key Files**: 
  - `algo.py`: Core algorithm implementation
  - `executor.py`: Execution management
  - `runner.py`: Graph runner
  - `loop.py`: Main execution loop

#### 2. Graph System (`langgraph/graph/`)
- **Purpose**: Graph definition and state management
- **Key Components**:
  - State management (stateful workflows)
  - Message passing between nodes
  - Branch logic and conditional execution

#### 3. Channels (`langgraph/channels/`)
- **Purpose**: Communication primitives between graph nodes
- **Types**: any_value, last_value, topic, etc.

#### 4. Checkpointing (`checkpoint/`, `checkpoint-*/`)
- **Purpose**: Persistence and state recovery
- **Implementations**: Memory, SQLite, PostgreSQL
- **Features**: Time travel, human-in-the-loop, durable execution

### Platform Components

#### LangGraph CLI (`libs/cli/`)
- **Purpose**: Development server and deployment tools
- **Key Features**:
  - Local development server
  - Docker integration
  - Configuration management
  - Template generation

#### SDKs (`sdk-py/`, `sdk-js/`)
- **Purpose**: Client libraries for LangGraph Platform
- **Features**: API clients, streaming, authentication

### Configuration

#### Main Config Files
- **pyproject.toml**: Python package configuration (in each lib)
- **langgraph.json**: LangGraph-specific configuration for graphs
- **uv.lock**: Dependency lock files
- **Makefile**: Build automation

#### Key Dependencies
- **Core**: langchain-core, pydantic, xxhash
- **Development**: pytest, mypy, ruff, jupyter
- **Optional**: psycopg (PostgreSQL), various AI model integrations

## Development Guidelines

### Code Style
- **Python**: Google-style docstrings, type hints required
- **Line Length**: 88 characters
- **Import Organization**: Handled by ruff
- **Type Checking**: Strict mypy configuration

### Testing Patterns
- Extensive use of fixtures for test data
- Snapshot testing with syrupy
- Docker compose for integration testing
- Mock objects for external dependencies

### Documentation Standards
- **Framework**: Diataxis (Tutorials, How-tos, References, Explanations)
- **Format**: Markdown + Jupyter notebooks
- **Style**: Concise, example-driven, heavily cross-linked
- **API Docs**: Auto-generated from docstrings

### Common Development Tasks

#### Adding a New Feature
1. **Design**: Start with discussion/issue for new features
2. **Implementation**: Create in appropriate `libs/` directory
3. **Tests**: Add comprehensive tests with fixtures
4. **Documentation**: Add to appropriate docs section
5. **Integration**: Update related components if needed

#### Running Specific Tests
```bash
# Run specific test file
make test TEST=tests/test_specific.py

# Run tests with pattern
make test TEST=tests/test_pregel.py::TestPregelAsync

# Run without Docker (if unavailable)
NO_DOCKER=true make test
```

#### Working with Documentation
```bash
# Build docs locally
cd docs
make serve-docs

# Lint docs
make codespell

# Execute notebooks (for testing)
bash execute_notebooks.sh
```

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `make install` or `uv sync`
2. **Docker Issues**: Check Docker is running, or use `NO_DOCKER=true`
3. **Test Failures**: Ensure PostgreSQL container is running
4. **Import Errors**: Check package is installed in editable mode

### Environment Setup
- **Virtual Environment**: Use `uv venv` or standard Python venv
- **Editable Installs**: Libraries are installed in editable mode for development
- **Path Dependencies**: Local packages reference each other via relative paths

### Performance Notes
- Use `make benchmark` for performance testing
- Profile with `make profile GRAPH=path/to/graph.py`
- Parallel testing available with `make test_parallel`

## Key Contacts & Resources

- **Repository**: https://github.com/langchain-ai/langgraph
- **Documentation**: https://langchain-ai.github.io/langgraph/
- **Issues**: Use GitHub issues for bugs, discussions for features
- **Contributing**: See CONTRIBUTING.md for detailed guidelines
- **License**: MIT

## Quick Reference

### File Extensions & Purposes
- `.py`: Python source code
- `.ipynb`: Jupyter notebooks (examples and docs)
- `.md`: Markdown documentation
- `.toml`: Configuration files (pyproject.toml)
- `.json`: Configuration (langgraph.json, package.json)
- `.lock`: Dependency locks (uv.lock, yarn.lock)

### Important Directories to Know
- `libs/langgraph/langgraph/`: Core framework code
- `docs/docs/`: All user-facing documentation
- `examples/`: Development examples (not for docs)
- `tests/`: Test suites with extensive fixtures
- `libs/cli/`: Command-line tools
- `libs/checkpoint*/`: Persistence implementations

This is a sophisticated, production-ready framework with enterprise-grade features like durable execution, human-in-the-loop capabilities, and comprehensive streaming support.