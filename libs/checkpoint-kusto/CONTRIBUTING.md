# Contributing to LangGraph Checkpoint Kusto

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Kusto cluster (for integration tests)
- Git

### Clone and Install

```bash
# Clone the repository
cd libs/checkpoint-kusto

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Environment Variables

For integration tests, set:

```bash
export KUSTO_CLUSTER_URI="https://your-cluster.region.kusto.windows.net"
export KUSTO_DATABASE="test_db"
export KUSTO_RUN_INTEGRATION="true"
```

## Development Workflow

### 1. Code Formatting

Run before committing:

```bash
make format
```

This runs:
- `ruff format` for code formatting
- `ruff check --fix` for auto-fixable linting issues

### 2. Linting

Check code quality:

```bash
make lint
```

This runs:
- `ruff check` for linting
- `mypy --strict` for type checking

### 3. Testing

Run tests:

```bash
# Unit tests only (no external dependencies)
make test-unit

# Integration tests (requires live Kusto cluster)
make test-integration

# All tests
make test

# Watch mode (auto-run on file changes)
make test-watch
```

### 4. Local Validation

Before submitting a PR, ensure:

```bash
# All checks pass
make format && make lint && make test-unit
```

## Code Standards

### Type Hints

All functions must have type hints:

```python
def my_function(arg1: str, arg2: int) -> dict[str, Any]:
    """Docstring here."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> dict[str, Any]:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When something goes wrong.
        
    Example:
        ```python
        result = my_function("test", 42)
        ```
    """
    pass
```

### Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(
    "Operation completed",
    extra={
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
        "duration_ms": duration,
    },
)
```

### Error Handling

Provide context in exceptions:

```python
try:
    result = await client.execute(query)
except Exception as e:
    logger.error(
        "Query failed",
        extra={"query": query, "error": str(e)},
    )
    raise ValueError(f"Failed to execute query: {e}") from e
```

## Testing Guidelines

### Unit Tests

- Fast (<1ms per test)
- No external dependencies
- Mock Kusto clients
- Focus on logic and edge cases

```python
@pytest.mark.unit
def test_version_increment():
    saver = BaseKustoSaver()
    v1 = saver.get_next_version(None, None)
    v2 = saver.get_next_version(v1, None)
    assert int(v2.split(".")[0]) == int(v1.split(".")[0]) + 1
```

### Integration Tests

- Require live Kusto cluster
- Env-gated (skip if not configured)
- Clean up resources after test
- Handle eventual consistency

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_roundtrip():
    async with _test_saver() as saver:
        # Test logic
        await saver.flush()
        await asyncio.sleep(1)  # Wait for ingestion
        # Verify
```

### Test Data

Use the `test_data` fixture from conftest.py:

```python
def test_something(test_data):
    config = test_data["configs"][0]
    checkpoint = test_data["checkpoints"][0]
    # ...
```

## Pull Request Process

### 1. Branch Naming

Use descriptive branch names:

- `feature/add-retry-logic`
- `fix/ingestion-timeout`
- `docs/update-readme`
- `test/add-load-tests`

### 2. Commit Messages

Follow conventional commits:

```
feat: add retry logic with exponential backoff

- Add RetryPolicy class
- Implement exponential backoff
- Add configuration options
- Update tests

Closes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

### 3. PR Checklist

Before submitting:

- [ ] Code formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Tests pass (`make test-unit`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Type hints added
- [ ] Docstrings added

### 4. PR Description

Include:

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [x] Tests pass
- [x] Documentation updated
- [x] CHANGELOG updated
```

## Architecture Guidelines

### Adding New Methods

1. Add to `BaseKustoSaver` if shared logic
2. Implement in `AsyncKustoSaver`
3. Add sync wrapper if needed
4. Update type hints and docstrings
5. Add tests (unit + integration)
6. Update documentation

### Modifying Queries

1. Test KQL syntax in Kusto Web UI first
2. Parameterize with `_build_kql_filter()`
3. Handle null/empty results
4. Add logging
5. Update tests

### Adding Configuration Options

1. Add parameter to `__init__()`
2. Document in docstring
3. Add to README configuration section
4. Add validation
5. Add test coverage

## Common Tasks

### Adding a New Feature

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes
# ...

# Test
make test-unit

# Commit
git add .
git commit -m "feat: add my feature"

# Push and create PR
git push origin feature/my-feature
```

### Fixing a Bug

```bash
# Create branch
git checkout -b fix/bug-description

# Write failing test first
# ...

# Fix the bug
# ...

# Verify test passes
make test-unit

# Commit
git commit -m "fix: resolve bug description"
```

### Updating Documentation

```bash
# Update relevant files
# - README.md for user docs
# - CHANGELOG.md for changes
# - Inline docstrings for API docs

# Verify markdown
# Check for broken links, formatting

# Commit
git commit -m "docs: update XYZ documentation"
```

## Release Process

Maintainers only:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create git tag: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. Build: `make build`
6. Publish: `make publish`

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with proposal
- **Security**: See SECURITY.md

## Code of Conduct

Be respectful and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
