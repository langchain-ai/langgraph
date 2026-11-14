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

Run Agent Server in development mode with hot reloading.

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

Launch Agent Server in Docker.

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

### `langgraph deploy`

Build and deploy an Agent Server image to LangSmith. Use `--agent-server-id` to
update an existing Agent Server and `--agent-server-type` to select its type.
The legacy `--deployment-id` and `--deployment-type` spellings remain supported
as hidden compatibility aliases, with no removal scheduled. Older Python CLI
versions and the JavaScript CLI may still require the legacy spellings; check
command help for your installed version.

#### Where `deploy` sends requests

All `deploy` commands need only a LangSmith API key (`LANGSMITH_API_KEY`). The CLI
asks the LangSmith API (`LANGSMITH_ENDPOINT`, or `https://api.smith.langchain.com`
by default) which workspaces the key can use. If the workspace runs on a
LangSmith data plane in your own cloud (BYOC on AWS, Azure, or GCP), the CLI
talks to that data plane's host API and sets the workspace for you. The data
plane URL must use https, since the API key is sent there. Status links still
point at the LangSmith UI. If the key's workspaces sit on different data planes,
set `LANGSMITH_TENANT_ID` to the workspace ID to reach its data plane; without
it, the CLI starts on the LangSmith control plane, and the workspace ID you
enter when asked picks the data plane. A `LANGSMITH_TENANT_ID` the key can't
use is an error. If the workspace lookup fails, the CLI says so and uses the
LangSmith control plane.

`--host-url` (or `LANGGRAPH_HOST_URL`) sends every request to the given host
backend URL instead and skips this lookup.

#### What a data plane supports

On a BYOC data plane, the data plane decides which deployments `deploy` can make:

- Building with `deploy` (a local Docker build, or `--remote`) needs a data plane
  that supports CLI builds. Otherwise, push to a registry you manage with
  `--push-to <registry>/<repository>`, or deploy an existing image with
  `--image-uri`. Both work on every data plane.
- `--agent-server-type` defaults to `serverless`, or to `dev` on a data plane
  without scale to zero. Asking for `serverless` or `dedicated` on a data plane
  that doesn't support it fails before anything is created.

### `langgraph deploy metrics`

Fetch Agent Server monitoring time series as JSON. Like the other `deploy`
commands, it finds the API key's workspace and sends the request to that
workspace's control plane or data plane (see
[Where `deploy` sends requests](#where-deploy-sends-requests)). Use
`langgraph deploy list` to find the Agent Server ID.

```bash
langgraph deploy metrics AGENT_SERVER_ID \
  --start-time 2026-10-28T00:00:00Z \
  --end-time 2026-10-28T01:00:00Z \
  --metrics cpu_usage,memory_usage,worker_count
```

`--metrics` accepts a comma-separated list of these names (also listed in
`langgraph deploy metrics --help`):

- Resource usage: `memory_usage`, `cpu_usage`, `disk_usage`
- Counts: `restart_count`, `replica_count`, `worker_count`, `lg_run_count`
- HTTP traffic: `responses_per_second`, `error_responses_per_second`, `p95_latency`
- Queue: `run_queue_wait_time`

These are exact metric names, not free-text search terms. Availability depends on
the Agent Server and platform; a supported metric may have no data. Use
`lg_run_count` for Agent Server runs, not the UI schema's placeholder `run_count`.

Omit `--metrics` to fetch all available metrics. Time bounds default to the last
15 minutes, and the backend picks the resolution from the time
range. This command never prompts; for an organization-scoped key, set
`LANGSMITH_TENANT_ID` to the workspace ID.

Default output is compact JSON grouped by section title, chart title, and series
name. Each series contains its real `project_metric` as `metric`, inferred `unit`
(`percent`, `count`, or `null`), `latest`, `min`, `max`, `avg`, and
`points: [[iso_timestamp, value], ...]`. Values are rounded to three decimals;
statistics use unrounded finite values and ignore missing samples. `latest` is
the latest non-missing value. Missing samples remain `null` gaps, and series with
no values have null statistics. Empty sections are marked `{"_empty":true}`.

The top level includes `deployment_id`, `start_time`, `end_time`, and
`step_seconds`. Bounds always describe the requested window, not the observed
point bounds. The CLI resolves omitted bounds once: end defaults to now (UTC),
and start defaults to 15 minutes before the resolved end. With only `--end-time`,
start is 15 minutes before that value; with only `--start-time`, end is now. Both
values are sent explicitly to the backend and output, even when there are no points. `step_seconds` is the most common gap between consecutive
timestamps within series, with ties broken by the smallest gap. Whole seconds
are integers; fractional seconds are preserved. It is `null` only when every
series has fewer than two timestamps. UI metadata and repeated series IDs are
omitted. The host-backend response itself is unchanged.

```bash
langgraph deploy metrics AGENT_SERVER_ID --summary
langgraph deploy metrics AGENT_SERVER_ID --raw
```

`--summary` returns the same shape without `points`. `--raw` returns the original
UI chart schema without reshaping or rounding. These flags are mutually exclusive.
Neither mode returns traces.

Size comparison using the realistic synthetic unit-test fixture (compact JSON
serialization for all modes; not a live Agent Server benchmark):

| Before: raw UI schema | After: default | After: summary |
| --- | --- | --- |
| 4,302 bytes | 1,226 bytes (72% smaller) | 806 bytes (81% smaller) |

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
