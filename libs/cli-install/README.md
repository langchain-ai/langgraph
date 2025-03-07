# LangGraph CLI Installer

A simple installer for the LangGraph CLI that uses `uv` to create an isolated environment.

## Why?

This lightweight installer creates an isolated installation of LangGraph CLI without worrying about Python environment conflicts or dependencies. It uses [uv](https://github.com/astral-sh/uv) to create a standalone environment with LangGraph CLI.

Key benefits:
- Prevents conflicts with other Python packages
- No knowledge of virtual environments needed
- Adds to your PATH automatically
- Installs the latest version of LangGraph CLI

## Quick Install

Simply run:

```bash
pip install langgraph-cli-install && langgraph-cli-install
```

This will:
1. Install the uv package if not already installed
2. Create an isolated environment with the latest LangGraph CLI
3. Add the CLI to your PATH automatically

After installation, you can run `langgraph --help` to get started.

## How It Works

This installer is similar to [aider-install](https://github.com/paul-gauthier/aider/blob/main/aider_install/main.py). It:

1. Uses the `uv` Python installer to create an isolated environment
2. Installs the latest `langgraph-cli` in that environment
3. Adds the installed binary to your PATH

This approach dramatically reduces installation issues caused by Python environment conflicts.

## Manual Installation

If you prefer not to use this installer, you can install LangGraph CLI directly:

```bash
pip install langgraph-cli
```

## License

MIT