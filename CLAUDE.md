# LangGraph Coding Guide

## Build/Test/Lint Commands

- Run all tests: `make test`
- Run single test: `make test TEST=path/to/test_file.py::test_function`
- Watch mode tests: `make test_watch`
- Run tests in parallel: `make test_parallel`
- Generate coverage report: `make coverage`
- Format code: `make format`
- Lint code: `make lint`
- Check spelling: `make spell_check`
- Fix spelling: `make spell_fix`
- Build documentation: `make serve-docs` (from repo root)
- Run benchmarks: `make benchmark` or `make benchmark-fast`

## Code Style Guidelines

- Follow [ruff](https://github.com/astral-sh/ruff) formatting/linting rules
- Use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings
- Enforce type annotations with mypy (`disallow_untyped_defs = True`)
- Use double quotes for strings
- Maximum line length of 88 characters
- Follow imports sorting with `ruff`
- All functions/classes must have proper docstrings with args/returns
- Write comprehensive unit tests for new features
- Keep backward compatibility
- PR scope should be isolated (changes shouldn't affect multiple packages)
- Use descriptive variable names following Python conventions
- Error handling should use appropriate exception types and messaging

## Feature Overview

langgraph is an orchestration framework (in the style of airflow or temporal) designed for LLM applications, with a focus on streaming output, cyclical and parallel workflows, and interrupt/resume capabilities. Applications built with langgraph are variously called workflows, graphs, cognitive architectures, agents. Key features:

1. **Graph-based Architecture**: Build directed computation graphs with nodes and edges
2. **State Management**: Type-safe state schema with custom reducers and transformations
3. **Human-in-the-loop**: Support for interrupts, checkpoints, and tool call review
4. **Persistence**: Save and resume execution with in-memory or database storage
5. **Streaming**: Multiple modes (values, updates, custom) for real-time feedback
6. **Multi-agent Patterns**: Support for network, supervisor, and hierarchical architectures

## Repository Structure

LangGraph follows a monorepo organization, with the following structure:

- `docs/` contains the source code (markdown and jupyter notebooks) for our documentation (hosted at https://langchain-ai.github.io/langgraph/)
- `libs/langgraph` is the main library, published to pypi as `langgraph`. This contains the majority of the code for the framework, as well as the majority of the unit tests.
- `libs/checkpoint` , published to pypi as `langgraph-checkpoint` contains the base classes for the persistence layer of langgraph. The two main abstractions are BaseCheckpointSaver (base class for persistence of workflow runs step-by-step) and BaseStore (base class for "long-term memory" operations, offering a key-value interface combined with semantic search over documents, used for persisting information across distinct workflow runs). This library is a dependency of both the main langgraph library, as well as implementations of these storage interfaces for specific databases. This library also contains reference implementations
- `libs/checkpoint-postgres` published to pypi as langgraph-checkpoint-postgres, contains implementations of checkpoint and store backed by postgres. Majority of the test coverage is in `libs/langgraph` in the form of tests that run over all storage implementations in the repo.
