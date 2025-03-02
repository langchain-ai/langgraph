# LangGraph Coding Guide

## Repository Structure

LangGraph follows a monorepo organization, with the following structure:

- `libs/langgraph` is the main Python library, published to pypi as `langgraph`. This contains the majority of the code for the framework, as well as the majority of the unit tests.
- `libs/checkpoint` , published to pypi as `langgraph-checkpoint` contains the base classes for the persistence layer of langgraph. The two main abstractions are BaseCheckpointSaver (base class for persistence of workflow runs step-by-step) and BaseStore (base class for "long-term memory" operations, offering a key-value interface combined with semantic search over documents, used for persisting information across distinct workflow runs). This library is a dependency of both the main langgraph library, as well as implementations of these storage interfaces for specific databases. This library also contains reference implementations
- `libs/checkpoint-postgres` published to pypi as langgraph-checkpoint-postgres, contains implementations of checkpoint and store backed by postgres. Majority of the test coverage is in `libs/langgraph` in the form of tests that run over all storage implementations in the repo.
- `langgraph-java` contains a Java implementation of the langgraph framework, which is in the early stages of development.

## Feature Overview

langgraph is an orchestration framework (in the style of airflow or temporal) designed for LLM applications, with a focus on streaming output, cyclical and parallel workflows, and interrupt/resume capabilities. Applications built with langgraph are variously called workflows, graphs, cognitive architectures, agents. Key features:

1. **Graph-based Architecture**: Build directed computation graphs with nodes and edges
2. **State Management**: Type-safe state schema with custom reducers and transformations
3. **Human-in-the-loop**: Support for interrupts, checkpoints, and tool call review
4. **Persistence**: Save and resume execution with in-memory or database storage
5. **Streaming**: Multiple modes (values, updates, custom) for real-time feedback
6. **Multi-agent Patterns**: Support for network, supervisor, and hierarchical architectures

## Python Development

### Build/Test/Lint Commands

(in the respective subdirectory)

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

### Code Style Guidelines

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

## Java Development

(in the `langgraph-java` subdirectory)

### Build/Test/Lint Commands

- Build the project: `./gradlew build`
- Run tests: `./gradlew test`
- Run a specific test: `./gradlew test --tests "com.langgraph.package.TestClass.testMethod"`
- Check formatting: `./gradlew spotlessCheck`
- Apply formatting: `./gradlew spotlessApply`
- Run all checks: `./gradlew check`
- Generate Javadoc: `./gradlew javadoc`

### Code Style Guidelines

- Follow standard Java code style (Google Java Style Guide)
- Use 4 spaces for indentation
- Maximum line length of 100 characters
- All public methods/classes must have proper Javadoc with @param/@return tags
- Use descriptive variable names following Java conventions (camelCase)
- Exception handling should use appropriate exception types with descriptive messages
- Favor composition over inheritance
- Use the Builder pattern for complex object creation
- Write comprehensive unit tests for new features

### Python Compatibility Guidelines

- When implementing features from the Python version:
  - Maintain semantic equivalence with the Python implementation
  - Preserve the same behavior for all public APIs
  - Document any intentional differences in behavior with comments
  - Pay special attention to collections handling (Python lists vs Java Lists)
  - Ensure that iteration order and value handling match Python where relevant
- Use the same test cases as the Python version when possible
- Do not introduce Java-specific shortcuts that would break Python compatibility
- Never add test-specific code to source files - tests should adapt to implementation, not vice versa

### Implementation Mapping

- Always consult and update the `PYTHON_JAVA_MAPPING.md` file when:
  - Adding new Java files or classes
  - Updating existing Java implementations
  - Fixing test failures in Java
  - Implementing Python features in Java
- This mapping file documents:
  - Where to find equivalent functionality in Python and Java
  - Any intentional deviations between implementations
  - Implementation status and compatibility notes
- When tests fail, check if the Java implementation matches Python behavior:
  - Fix the implementation to match Python semantics whenever possible
  - Update tests only if the Python version also differs
  - Never create special cases or workarounds in Java just to make tests pass
  - Document any implementation differences clearly in the mapping file
- For new features, implement the Python behavior first, then adapt to Java idioms

### Backward Compatibility and API Design

- LangGraph Java has not been released publicly, so there is no need to maintain backward compatibility
- When renaming methods, members, or classes:
  - Use the clearest, most intuitive names that match Python semantics
  - Remove old/deprecated methods completely rather than marking them as deprecated
  - Update all tests and documentation to use the new names
  - Do not leave deprecated methods or tests for backward compatibility

### API Design Principles

- Prefer a single, clear way to accomplish each task rather than multiple convenience methods
- Prefer builder patterns over static factory methods where appropriate
- For collections, prefer methods that operate on collections rather than having both single-item and collection variants
- Choose method names that clearly express their purpose and align with Java conventions
- Maintain consistent naming patterns across similar components
- Document the recommended usage pattern in JavaDoc

### Project Structure

- `langgraph-core`: Core functionality of the framework
- `langgraph-checkpoint`: Persistence layer for checkpoints and state management
- `langgraph-examples`: Example applications and usage patterns

### Error Handling

- Use runtime exceptions for unexpected errors
- Use checked exceptions for recoverable errors
- Provide clear error messages that include context about what went wrong
- Validate inputs early to prevent cascading errors
- Ensure all resources are properly closed even in error conditions
