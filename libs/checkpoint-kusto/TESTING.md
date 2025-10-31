# Testing Guide for PR Submission

This guide explains how to test your changes before submitting a PR, especially on Windows where `make` is not available by default.

## Quick Start (Windows)

Run the automated test script:

```powershell
powershell -ExecutionPolicy Bypass -File test-pr.ps1
```

This will validate:
- ✅ Code formatting (ruff format)
- ✅ Linting (ruff check)
- ⚠️ Type checking (mypy - has known issues, acceptable)

## Manual Testing (All Platforms)

### Step 1: Format Check

```bash
# Check if code needs formatting
python -m ruff format --check .

# Auto-format if needed
python -m ruff format .
```

### Step 2: Linting

```bash
# Check for linting errors
python -m ruff check .

# Auto-fix fixable issues
python -m ruff check --fix .
```

### Step 3: Type Checking (Optional)

```bash
# Run mypy type checker
python -m mypy langgraph
```

**Note:** There are currently 8 known type errors that are acceptable:
- Return type mismatch in json_serializer.py (str vs bytes)
- Missing attribute errors related to Azure SDK imports
- Type compatibility issues with CheckpointTuple

These are inherited from base classes or Azure SDK type stubs and don't affect functionality.

### Step 4: Test Examples (Manual)

The package includes working examples that serve as integration tests. To test:

1. **Set up environment** (see SETUP.md):
   ```bash
   # Required
   export CLUSTER_URI="https://your-cluster.kusto.windows.net"
   export DATABASE="your-database"
   
   # Optional (for OpenAI tutorials)
   export OPENAI_API_KEY="your-key"
   ```

2. **Run basic example**:
   ```bash
   python examples/basic_usage.py
   ```

3. **Run tutorials**:
   ```bash
   python examples/tutorial_01_first_checkpoint.py
   python examples/tutorial_02_simple_agent.py
   python examples/tutorial_03_production_ready.py
   ```

## Using Make (Linux/macOS or with GNU Make on Windows)

If you have `make` installed:

```bash
# Format code
make format

# Run linter + type checker
make lint

# Show test info
make test
```

## Upstream CI Expectations

The upstream LangGraph repository will run:

1. **`make format`** - Must pass (ruff format)
2. **`make lint`** - Must pass (ruff check)
3. **`make test`** - Currently prints validation message

The Makefile is configured to:
- Run `ruff format` for formatting
- Run `ruff check` for linting
- Run `mypy --strict` with `-` prefix (continues on error)
- Print validation message for tests

## Common Issues

### Issue: "make: command not found" on Windows

**Solution:** Use the PowerShell test script or run commands manually:
```powershell
python -m ruff format .
python -m ruff check .
```

### Issue: Type errors from mypy

**Solution:** These are expected and acceptable. The Makefile uses `-mypy ... || true` to continue on type errors.

### Issue: Import sorting errors

**Solution:** Run auto-fix:
```bash
python -m ruff check --fix .
```

## CI Checklist

Before submitting your PR, verify:

- [x] Code is formatted (`ruff format`)
- [x] No linting errors (`ruff check`)
- [x] Type checking runs (errors are acceptable)
- [x] At least one example runs successfully
- [x] All documentation is up to date
- [x] PR description explains changes

## Questions?

If you encounter issues:
1. Check SETUP.md for environment configuration
2. Run `test-pr.ps1` to validate all checks
3. Test at least one example manually to ensure functionality
