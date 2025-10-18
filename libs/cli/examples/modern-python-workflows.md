# Modern Python Dependency Manager Support

This example demonstrates how `langgraph dev` now supports modern Python dependency managers like pipenv, poetry, and uv.

## Supported Dependency Managers

### 1. Pipenv
Create a `Pipfile` in your project root:
```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
langchain-openai = "*"
langchain-community = "*"

[dev-packages]
pytest = "*"
```

Then run:
```bash
pipenv install
pipenv run langgraph dev
```

### 2. Poetry
Create a `pyproject.toml` with Poetry configuration:
```toml
[tool.poetry]
name = "my-langgraph-app"
version = "0.1.0"
description = "A LangGraph application"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
langchain-openai = "^0.1.0"
langchain-community = "^0.2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Then run:
```bash
poetry install
poetry run langgraph dev
```

### 3. uv
Create a `pyproject.toml` with uv configuration:
```toml
[project]
name = "my-langgraph-app"
version = "0.1.0"
description = "A LangGraph application"
requires-python = ">=3.11"
dependencies = [
    "langchain-openai>=0.1.0",
    "langchain-community>=0.2.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
]
```

Then run:
```bash
uv sync
uv run langgraph dev
```

## Automatic Detection

The CLI automatically detects which dependency manager you're using based on the presence of:

- `Pipfile` → Pipenv
- `pyproject.toml` with `[tool.poetry]` → Poetry  
- `uv.lock` or `requirements.lock` → uv
- `pyproject.toml` with `[tool.uv]` → uv

## Virtual Environment Activation

When you run `langgraph dev`, the CLI will:

1. **Detect** your dependency manager
2. **Activate** the corresponding virtual environment
3. **Add** the virtual environment's packages to Python's import path
4. **Display** a confirmation message

Example output:
```
✅ Detected poetry project and activated virtual environment
```

## Automatic Dependency Installation

Use the `--install-deps` flag to automatically install dependencies:

```bash
langgraph dev --install-deps
```

This will:
1. Detect your dependency manager
2. Run the appropriate install command (`pipenv install`, `poetry install`, `uv pip install -e .`)
3. Activate the virtual environment

## Fallback Behavior

If virtual environment activation fails, the CLI will:
- Display a warning message
- Fall back to using system Python
- Continue with the development server

## Benefits

- **No more import errors** when using modern dependency managers
- **Automatic virtual environment detection** and activation
- **Consistent experience** across different Python workflows
- **Backward compatibility** with existing pip-based projects
