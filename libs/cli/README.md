# LangGraph CLI

[![PyPI - Version](https://img.shields.io/pypi/v/langgraph-cli?label=%20)](https://pypi.org/project/langgraph-cli/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langgraph-cli)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langgraph-cli)](https://pypistats.org/packages/langgraph-cli)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

To help you ship LangGraph apps to production faster, check out [LangSmith](https://www.langchain.com/langsmith).
[LangSmith](https://www.langchain.com/langsmith) is a unified developer platform for building, testing, and monitoring LLM applications.

## Quick Install

```bash
uv add langgraph-cli
```

## 🤔 What is this?

The LangGraph CLI is the official command-line interface for LangGraph. It provides tools to create, develop, build, and run LangGraph applications locally or in Docker.

## 📖 Documentation

For full documentation, see the [LangGraph CLI reference](https://reference.langchain.com/python/langgraph-cli). For conceptual guides and tutorials, see the [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview).

For development mode with hot reloading:

```bash
uv add "langgraph-cli[inmem]"
```

## Commands

### `langgraph new` 🌱

Create a new LangGraph project from a template.

```bash
langgraph new [PATH] --template TEMPLATE_NAME
```

### `langgraph dev` 🏃‍♀️

Run LangGraph API server in development mode with hot reloading.

```bash
langgraph dev [OPTIONS]
  --host TEXT                 Host to bind to (default: 127.0.0.1)
  --port INTEGER             Port to bind to (default: 2024)
  --no-reload               Disable auto-reload
  --debug-port INTEGER      Enable remote debugging
  --no-browser             Skip opening browser window
  -c, --config FILE        Config file path (default: langgraph.json)
```

### `langgraph up` 🚀

Launch LangGraph API server in Docker.

```bash
langgraph up [OPTIONS]
  -p, --port INTEGER        Port to expose (default: 8123)
  --wait                   Wait for services to start
  --watch                  Restart on file changes
  --verbose               Show detailed logs
  -c, --config FILE       Config file path
  -d, --docker-compose    Additional services file
```

### `langgraph build`

Build a Docker image for your LangGraph application.

```bash
langgraph build -t IMAGE_TAG [OPTIONS]
  --platform TEXT          Target platforms (e.g., linux/amd64,linux/arm64)
  --pull / --no-pull      Use latest/local base image
  -c, --config FILE       Config file path
```

### `langgraph dockerfile`

Generate a Dockerfile for custom deployments.

```bash
langgraph dockerfile SAVE_PATH [OPTIONS]
  -c, --config FILE       Config file path
```

### `langgraph deploy metrics`

Fetch deployment monitoring time series as JSON through the LangSmith control plane.
Use `langgraph deploy list` to find the deployment ID. Authentication uses the same
API key and workspace configuration as the other deployment commands.

```bash
langgraph deploy metrics DEPLOYMENT_ID \
  --start-time 2026-10-28T00:00:00Z \
  --end-time 2026-10-28T01:00:00Z \
  --step-seconds 60 \
  --metrics cpu_usage,memory_usage,worker_count
```

Omit `--metrics` to fetch all available metrics. Time bounds and resolution default
to the backend's settings (start of the current hour through now). Output contains
charts and their time-series data, not traces. This command never prompts; for an
organization-scoped key, set `LANGSMITH_TENANT_ID` to the workspace ID.

### `langgraph deploy scale`

Change a deployment's compute tier, dedicated Postgres tier, or both. Omitted
settings remain unchanged. The backend enforces deployment permissions, supported
tiers, and platform restrictions.

```bash
langgraph deploy scale DEPLOYMENT_ID \
  --compute-tier DEDICATED_M --database-tier DEDICATED_L
```

Compute tiers are `SERVERLESS_S`, `SERVERLESS_M`, `SERVERLESS_L`, `DEDICATED_S`,
`DEDICATED_M`, and `DEDICATED_L`. Database tiers are available only for dedicated
deployments: `DEDICATED_S`, `DEDICATED_M`, and `DEDICATED_L`. Availability depends
on the deployment's platform and enabled features.

Scaling prompts for confirmation because it creates a revision, may interrupt
service, and may change billing. Pass `--yes` to confirm without prompting, with
credentials and workspace configured for non-interactive use. The command reports
that scaling was requested; it does not wait for the revision to become healthy.
Arbitrary worker replica overrides are not supported by this command.

## Configuration

The CLI uses a `langgraph.json` configuration file with these key settings:

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_graph": "./your_package/file.py:graph"
  },
  "env": "./.env",
  "python_version": "3.11",
  "pip_config_file": "./pip.conf",
  "dockerfile_lines": []
}
```

Git dependencies should use credential-free URLs. The CLI conservatively scans direct `langgraph.json` dependencies, common Python package files, uv project and lock files, and common Node.js package and lock files for HTTP Git URLs with userinfo. This check is not exhaustive: generated Docker builds can copy other files, including nested requirement or constraint files, into image layers without scanning them. For private dependencies, provide short-lived credentials through your build environment's secret-backed Git credential helper. Do not store credentials in copied files such as `langgraph.json` or `pip_config_file`.

See the [full documentation](https://reference.langchain.com/python/langgraph-cli) for detailed configuration options.

## Development

To develop the CLI itself:

1. Clone the repository
2. Navigate to the CLI directory: `cd libs/cli`
3. Install development dependencies: `uv sync`
4. Make your changes to the CLI code
5. Test your changes:

```bash
# Run CLI commands directly
uv run langgraph --help

# Or use the examples
cd examples
uv sync
uv run langgraph dev  # or other commands
```

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
