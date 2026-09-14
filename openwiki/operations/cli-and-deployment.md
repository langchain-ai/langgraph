---
type: Operations
title: CLI and Deployment
description: LangGraph CLI commands for development, building, and deployment workflows, with Docker Compose configuration, environment setup, and production deployment patterns.
tags: [cli, deployment, docker, development, build, configuration, operations]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:19:56.419Z
sources:
  - id: openwiki-source-267e2c951319a772291005e0
    resource: repo://libs/cli/langgraph_cli/cli.py
  - id: openwiki-source-4070aba88b5371eb9288b448
    resource: repo://libs/cli/langgraph_cli/config.py
  - id: openwiki-source-8ddf3a1ffe6952f2ed9960e7
    resource: repo://libs/cli/langgraph_cli/deploy.py
  - id: openwiki-source-828c489ee3636c0f44019790
    resource: repo://libs/cli/langgraph_cli/docker.py
  - id: openwiki-source-abee9b657b25f66cb7823ffb
    resource: repo://libs/cli/langgraph_cli/schemas.py
  - id: openwiki-source-7fdc76d44cc41fb11e82a910
    resource: repo://libs/cli/langgraph_cli/templates.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:19:56.419Z" }
---

## Overview

The **LangGraph CLI** is the command-line interface for scaffolding, developing, building, and deploying LangGraph applications. It provides a unified workflow from local development (`langgraph dev`) through containerized deployment (`langgraph up`, `langgraph build`, `langgraph deploy`) with built-in support for Docker Compose orchestration, hot-reload development, multi-platform image builds, and remote deployment to LangSmith.

The CLI discovers graphs from Python or JavaScript modules—either explicitly listed in `langgraph.json` as `path:object_name` references or automatically via `@entrypoint` decorated functions—and manages their dependencies, environment, and runtime configuration.

---

## Core Commands

### `langgraph new` — Project Scaffolding

Creates a new LangGraph project from a template:

```bash
langgraph new [PATH] --template TEMPLATE_NAME
```

**Templates:**
- `new-langgraph-project-python` — Minimal chatbot with memory (Python)
- `new-langgraphjs-project` — Minimal chatbot with memory (JavaScript)
- `simple-agent-template` — Flexible agent with tool extension (Python)
- `deep-agent-template-python` / `deep-agent-template-js` — Opinionated deep agent template

The command downloads the template repository from GitHub, extracts it, and customizes it for your project name. It initializes a `langgraph.json` config file with scaffolded graph definitions, dependencies, and environment variable placeholders.

**Evidence:** [`repo://libs/cli/langgraph_cli/templates.py#L10-L40`](repo://libs/cli/langgraph_cli/templates.py#L10-L40)

### `langgraph dev` — Development Mode

Runs the LangGraph API server in-memory with hot reloading and debugging:

```bash
langgraph dev [OPTIONS]
  --host TEXT                      Network interface (default: 127.0.0.1)
  --port INTEGER                   Port (default: 2024)
  --config, -c FILE                Config file (default: langgraph.json)
  --no-reload                      Disable auto-reload on file changes
  --no-browser                     Skip opening browser UI
  --debug-port INTEGER             Enable remote debugging on port
  --wait-for-client                Block until debugger connects
  --tunnel                         Expose via public tunnel (Cloudflare)
  --studio-url TEXT                URL of LangGraph Studio instance
  --allow-blocking                 Don't raise errors for blocking I/O
  --ssl-certfile FILE              SSL certificate for HTTPS
  --ssl-keyfile FILE               SSL key for HTTPS
  --n-jobs-per-worker INTEGER      Concurrent jobs per worker (default: 10)
  --server-log-level LEVEL         API server log level (default: WARNING)
```

The `dev` command loads graph definitions from `langgraph.json`, scans `dependencies` to add directories to `sys.path`, imports graph objects, and starts an in-process HTTP server. When `--no-reload` is not set, the server automatically restarts on file changes. When `--debug-port` is specified, the server listens for a remote debugger client (requires `debugpy`).

The `--tunnel` flag exposes the local server via a public Cloudflare tunnel, enabling browser access from networks that block `localhost`.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L663-L862`](repo://libs/cli/langgraph_cli/cli.py#L663-L862)

### `langgraph build` — Docker Image Build

Creates a Docker image for your LangGraph application:

```bash
langgraph build -t IMAGE_TAG [OPTIONS]
  -c, --config FILE                Config file (default: langgraph.json)
  --platform TEXT                  Target platforms (e.g., linux/amd64,linux/arm64)
  --base-image TEXT                Base image (defaults to langchain/langgraph-api)
  --api-version TEXT               Pin API server version
  --pull / --no-pull               Pull latest/use local base image
  --install-command TEXT           Custom install command (auto-detect by default)
  --build-command TEXT             Custom build command
  [DOCKER_BUILD_ARGS]              Pass-through args to docker build (e.g., --push)
```

The build process:
1. Validates the config file
2. Generates a Dockerfile from the config (Python version, dependencies, graphs, environment)
3. Optionally pulls the latest base image (LangGraph API server)
4. Runs `docker build` with the generated Dockerfile and any extra arguments

**Multi-Platform Support:** Use `--platform linux/amd64,linux/arm64` to build for multiple architectures simultaneously (requires Docker Buildx).

**Engine Runtime Mode:** Supports two modes:
- `combined_queue_worker` (default) — Single container runs both queue and worker threads
- `distributed` — Separate executor and orchestrator containers for horizontal scaling

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L380-L465`](repo://libs/cli/langgraph_cli/cli.py#L380-L465)

### `langgraph up` — Docker Compose Deployment

Launches the LangGraph server in Docker Compose with managed services (Postgres, Redis):

```bash
langgraph up [OPTIONS]
  -c, --config FILE                Config file (default: langgraph.json)
  -p, --port INTEGER               Expose port (default: 8123)
  -d, --docker-compose FILE        Additional services file
  --wait                           Wait for services to start
  --watch                          Restart on file changes
  --verbose                        Show detailed logs
  --pull / --no-pull               Pull latest/use local images
  --recreate / --no-recreate       Recreate containers
  --image TEXT                     Use pre-built image (skips build)
  --base-image TEXT                Base image for langgraph-api
  --postgres-uri TEXT              External Postgres (skips local DB)
  --debugger-port INTEGER          Enable debugger UI on port
  --debugger-base-url TEXT         Debugger API base URL
  --api-version TEXT               API server version
  --engine-runtime-mode MODE       combined_queue_worker or distributed
```

**Service Stack:**
- **PostgreSQL 16 with pgvector** — Checkpoint and vector store persistence. Default: local container (`langgraph-postgres` on port 5433). Port binding allows external access for debugging.
- **Redis** — Session and task queue. Default: local container (`langgraph-redis`).
- **LangGraph API Server** — Main application container. Bound to `--port`.
- **LangGraph Debugger** (optional) — Runs when `--debugger-port` is specified. Connects to Postgres and serves the debugger UI.

All services have health checks configured; the LangGraph API waits for Postgres and Redis to be healthy before starting.

**Volume Management:** When a local Postgres is created, state persists in the `langgraph-data` volume. Use `--postgres-uri` to connect to an external database and avoid managing local volume state.

**Environment Variables:** The server reads variables from:
- `.env` file (if referenced in config)
- System environment
- Config `env` dict

Required environment variables depend on deployment type:
- **Local dev**: `LANGSMITH_API_KEY` (for LangSmith integration)
- **Production**: `LANGGRAPH_CLOUD_LICENSE_KEY` (for self-hosted deployments)

**Health Checks:** Services report health via Docker health status. The LangGraph API health check runs `python /api/healthcheck.py` (Docker >= 25.0) to verify readiness; Postgres and Redis use native probes.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L242-L373`](repo://libs/cli/langgraph_cli/cli.py#L242-L373), [`repo://libs/cli/langgraph_cli/docker.py#L190-L301`](repo://libs/cli/langgraph_cli/docker.py#L190-L301)

### `langgraph dockerfile` — Dockerfile Generation

Generates a standalone Dockerfile and optional Docker Compose files:

```bash
langgraph dockerfile SAVE_PATH [OPTIONS]
  -c, --config FILE                Config file (default: langgraph.json)
  --add-docker-compose             Also generate docker-compose.yml and .env
  --base-image TEXT                Base image
  --api-version TEXT               API server version
```

Outputs:
- **Dockerfile** — Multi-stage build that installs dependencies and graph code
- **.dockerignore** (optional) — Excludes `.env`, node_modules, tests, etc.
- **docker-compose.yml** (optional) — Service definitions with Postgres, Redis
- **.env** (optional) — Environment variable template with comments

This command is useful for integrating LangGraph into custom CI/CD pipelines or when you need to customize the Dockerfile before building.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L527-L656`](repo://libs/cli/langgraph_cli/cli.py#L527-L656)

### `langgraph deploy` — Remote Deployment

Deploys to LangSmith or self-hosted LangGraph Cloud:

```bash
langgraph deploy [OPTIONS]
  --deployment-id TEXT             Deployment ID (or use --name)
  --name TEXT                      Deployment name (or use --deployment-id)
  --config, -c FILE                Config file (default: langgraph.json)
  --git-url TEXT                   Git repository URL
  --git-branch TEXT                Git branch (default: current)
  --git-commit TEXT                Git commit SHA
  --build-config TEXT              Build config JSON
  --include-all / --exclude-all    Include/exclude graphs (default: all)
  --graph TEXT                     Include specific graph(s)
  --json                           JSON output mode
```

The deploy flow:
1. Validates auth (requires `LANGSMITH_API_KEY` or `LANGGRAPH_HOST_API_KEY`)
2. Reads config and graphs from the repo (Git URL) or local file
3. Builds the Docker image (local or remote)
4. Uploads the image to the LangSmith registry
5. Polls deployment status until completion (or timeout)

**Build Strategies:**
- **Local build** — Uses local Docker daemon (faster for small images)
- **Remote build** — LangSmith builds in the cloud (useful for CI/CD without Docker)

**Graph Selection:**
- `--include-all` (default) — Deploy all graphs in config
- `--exclude-all` — Deploy none (useful with `--graph` to pick specific ones)
- `--graph NAME` — Include named graph(s); can be repeated

**Reserved Environment Variables:** The deploy command rejects certain variables (e.g., `POSTGRES_URI`, `LANGSMITH_API_KEY`) to prevent misconfiguration; these are reserved for the deployment infrastructure.

**Evidence:** [`repo://libs/cli/langgraph_cli/deploy.py#L1-L100`](repo://libs/cli/langgraph_cli/deploy.py#L1-L100)

### `langgraph validate` — Configuration Validation

Validates the config file syntax and graph references:

```bash
langgraph validate [OPTIONS]
  -c, --config FILE                Config file (default: langgraph.json)
```

Outputs the number of graphs found and any warnings about unknown keys.

---

## Configuration (`langgraph.json`)

The **langgraph.json** file declares your graph definitions, dependencies, environment, and build options.

### Minimal Example

```json
{
  "dependencies": ["langchain_anthropic", "."],
  "graphs": {
    "agent": "./src/agent.py:graph"
  }
}
```

### Full Example

```json
{
  "python_version": "3.11",
  "pip_installer": "auto",
  "dependencies": [
    "langchain_anthropic",
    "langchain_openai",
    "./src"
  ],
  "graphs": {
    "agent": {
      "path": "./src/agent.py:graph",
      "description": "Main conversational agent"
    },
    "research": {
      "path": "./src/research.py:research_graph",
      "description": "Research tool graph"
    }
  },
  "env": ".env",
  "store": {
    "indexer": "LangSmith"
  },
  "checkpointer": {
    "type": "postgres"
  },
  "auth": {
    "path": "./src/auth.py:get_auth_handler",
    "security": {
      "api_key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
  },
  "http": {
    "allowed_origins": ["https://example.com"],
    "routes": {
      "GET /custom/status": "./src/routes.py:status_handler"
    }
  },
  "webhooks": {
    "url_template": "https://example.com/webhooks/{event}",
    "authorization": {
      "header": "Authorization: Bearer ${WEBHOOK_SECRET}"
    }
  },
  "ui": {
    "chat_input": "./ui/ChatInput.tsx",
    "chat_output": "./ui/ChatOutput.tsx"
  },
  "keep_pkg_tools": false,
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y curl"
  ]
}
```

### Field Reference

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `python_version` | string | 3.11 | Python version for the container (3.11, 3.12, 3.13) |
| `node_version` | string | 20 | Node.js version for JavaScript graphs |
| `pip_installer` | string | auto | Package installer: `pip`, `uv`, or `auto` |
| `dependencies` | array | [] | Packages to install (PyPI, local paths, Git URLs) |
| `graphs` | object | {} | Graph definitions, key is assistant ID, value is `path:object` |
| `env` | string or object | {} | Environment variables (file path or key-value dict) |
| `store` | object | null | Long-term memory (vector indexing config) |
| `checkpointer` | object | null | Checkpoint persistence setup (type: postgres/sqlite/memory) |
| `auth` | object | null | Custom authentication (path to handler + OpenAPI security defs) |
| `encryption` | object | null | At-rest encryption setup (path to handler) |
| `http` | object | null | CORS origins and custom routes |
| `webhooks` | object | null | Outbound event delivery config |
| `ui` | object | null | Named custom UI components (JS/TS files) |
| `keep_pkg_tools` | bool/array | false | Retain Python packaging tools (pip, setuptools, wheel) |
| `dockerfile_lines` | array | [] | Additional Dockerfile lines (RUN, COPY, ENV, etc.) |
| `pip_config_file` | string | null | Path to pip.conf for auth-gated packages |
| `base_image` | string | null | Override base LangGraph API image |
| `api_version` | string | null | Pin API server version (e.g., `0.8`, `~=0.8.0.dev5`) |
| `image_distro` | string | debian | Base distro: `debian` or `wolfi` (lighter) |

**Evidence:** [`repo://libs/cli/langgraph_cli/schemas.py#L620-L774`](repo://libs/cli/langgraph_cli/schemas.py#L620-L774)

---

## Graph Discovery and Module Loading

The CLI discovers graph definitions by:

1. **Explicit References** — `path/to/file.py:object_name` strings in `graphs` dict
2. **@entrypoint Decorator** — Automatically detects `@entrypoint()` decorated functions

### Example: Explicit Graph Definition

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  }
}
```

Python code:
```python
from langgraph.graph import StateGraph

graph = StateGraph(...)
```

### Example: @entrypoint Decorator

Python code:
```python
from langgraph.func import entrypoint
from langgraph.types import RunnableConfig

@entrypoint()
def my_graph(state, config: RunnableConfig):
    return {"messages": [...]}
```

Config: The graph is auto-registered without explicit `langgraph.json` entry.

### Module Loading Flow

1. **Path Setup** — CLI adds `cwd` and directories in `dependencies` to `sys.path`
2. **Import** — Loads the Python/JavaScript module at the declared path
3. **Object Lookup** — Retrieves the named object (must be a Pregel/StateGraph or async context manager)
4. **Registration** — Server registers the graph as an assistant with the graph ID as the assistant ID

For context managers, the server invokes them with `RunnableConfig` to obtain the compiled graph at runtime.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L828-L835`](repo://libs/cli/langgraph_cli/cli.py#L828-L835)

---

## Docker Compose Configuration

The CLI generates Docker Compose files with interconnected services. You can customize this using a `-d, --docker-compose` file or by editing the generated `docker-compose.yml`.

### Default Service Network

```yaml
services:
  langgraph-postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: postgres
    volumes:
      - langgraph-data:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U postgres
      interval: 60s
      start_period: 10s

  langgraph-redis:
    image: redis:6
    healthcheck:
      test: redis-cli ping

  langgraph-api:
    image: langchain/langgraph-api:[VERSION]
    ports:
      - "8123:8000"
    environment:
      POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable
      REDIS_URI: redis://langgraph-redis:6379
    depends_on:
      langgraph-postgres:
        condition: service_healthy
      langgraph-redis:
        condition: service_healthy

volumes:
  langgraph-data:
    driver: local
```

### Adding Custom Services

Create a Docker Compose override file:

```yaml
# docker-compose.custom.yml
services:
  my-service:
    image: my-image
    environment:
      API_HOST: http://langgraph-api:8000

  custom-postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
```

Run with the override:
```bash
langgraph up -d docker-compose.custom.yml
```

### Environment Variables in Compose

Set variables in `.env` file (referenced in config):

```bash
# .env
LANGSMITH_API_KEY=your-api-key
OPENAI_API_KEY=sk-...
CUSTOM_VAR=value
```

Or set in `langgraph.json`:

```json
{
  "env": {
    "LANGSMITH_API_KEY": "your-api-key",
    "DEBUG": "true"
  }
}
```

**Evidence:** [`repo://libs/cli/langgraph_cli/docker.py#L190-L301`](repo://libs/cli/langgraph_cli/docker.py#L190-L301)

---

## Environment Variable Setup

### Sources (in priority order)

1. System environment variables
2. `.env` file (if `env` path specified in config)
3. Config `env` dict

### Common Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LANGSMITH_API_KEY` | LangSmith API authentication | `ls_...` |
| `LANGGRAPH_CLOUD_LICENSE_KEY` | Self-hosted deployment license | (production only) |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `POSTGRES_URI` | Postgres connection (auto-set in Docker Compose) | `postgresql://user:pass@host/db` |
| `REDIS_URI` | Redis connection (auto-set in Docker Compose) | `redis://host:6379` |
| `PORT` | Server port (default 8000, mapped to 8123 externally) | `8000` |
| `N_JOBS_PER_WORKER` | Max concurrent jobs per worker | `10` |

### Loading from .env File

```bash
# .env
LANGSMITH_API_KEY=sk-dev-...
OPENAI_API_KEY=sk-...
DEBUG=false
```

Reference in config:
```json
{
  "env": ".env"
}
```

The CLI parses and injects these into the container at runtime.

---

## Remote Debugging

### Development Mode Debugging

Enable remote debugging in `langgraph dev`:

```bash
langgraph dev --debug-port 5678 --wait-for-client
```

This:
1. Starts a debugger listener on port 5678
2. Blocks server startup until a debugger client connects
3. Allows you to step through graph execution in your IDE

**IDE Setup (VS Code):**
```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Attach to LangGraph Dev",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "127.0.0.1",
        "port": 5678
      },
      "logToFile": true
    }
  ]
}
```

### Docker Debugging

In `langgraph up`, enable the debugger UI:

```bash
langgraph up --debugger-port 3968
```

This launches a LangGraph Debugger container that serves a web UI at `http://localhost:3968`. The debugger connects to the Postgres database to inspect runs and graph state.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L143-L152`](repo://libs/cli/langgraph_cli/cli.py#L143-L152)

---

## Production Deployment Patterns

### Pattern 1: Docker Compose on a Single Server

Build and push image to registry:
```bash
langgraph build -t myregistry.azurecr.io/my-graph:latest --push
```

Deploy with external Postgres and custom environment:
```bash
langgraph up --image myregistry.azurecr.io/my-graph:latest \
  --postgres-uri postgresql://user:pass@external-host/db \
  --port 8000
```

Set environment variables in `.env`:
```bash
LANGGRAPH_CLOUD_LICENSE_KEY=license-key
LANGSMITH_API_KEY=api-key
```

### Pattern 2: Kubernetes Deployment

Generate Dockerfile:
```bash
langgraph dockerfile ./Dockerfile --add-docker-compose
```

Build and push image:
```bash
docker build -f ./Dockerfile -t myregistry/my-graph:v1.0 .
docker push myregistry/my-graph:v1.0
```

Deploy with Helm or Kustomize, using external Postgres/Redis.

### Pattern 3: LangSmith Cloud Deployment

Deploy to LangSmith (handles Postgres, Redis, load balancing):
```bash
langgraph deploy --name my-deployment \
  --git-url https://github.com/myorg/my-repo \
  --git-branch main
```

Or with local build:
```bash
langgraph deploy --name my-deployment \
  --config ./langgraph.json
```

LangSmith manages:
- Containerization and image storage
- PostgreSQL cluster for checkpoints
- Redis cluster for sessions
- TLS, load balancing, autoscaling
- Monitoring and log aggregation

---

## Port Configuration

### Default Ports

| Service | Port | Purpose |
|---------|------|---------|
| LangGraph API (local dev) | 2024 | `langgraph dev` server |
| LangGraph API (Docker) | 8123 | External port in `langgraph up` |
| LangGraph API (container) | 8000 | Internal port inside container |
| Postgres | 5433 | External access to database |
| Redis | (none) | Only accessible within Docker network |
| Debugger | (custom) | Set via `--debugger-port` |

### Changing Ports

**Development mode:**
```bash
langgraph dev --port 8000
```

**Docker Compose:**
```bash
langgraph up --port 9000
```

This maps the container port 8000 to host port 9000.

**Via config (not supported for CLI):** Ports must be set at runtime via CLI flags or Docker environment.

---

## Multi-Platform Builds

Build images for multiple architectures (e.g., for Apple Silicon and Linux servers):

```bash
langgraph build -t myregistry/my-graph:latest \
  --platform linux/amd64,linux/arm64 \
  --push
```

Requirements:
- Docker Buildx installed
- Registry credentials configured (`docker login`)
- `--push` to upload images

The build uses cross-compilation; each platform's image is built and pushed separately to the registry with the same tag.

**Evidence:** [`repo://libs/cli/langgraph_cli/docker.py#L52-L97`](repo://libs/cli/langgraph_cli/docker.py#L52-L97)

---

## Troubleshooting and Logging

### Verbose Logging

Enable detailed logs in Docker Compose:
```bash
langgraph up --verbose
```

This streams full output from all containers, useful for debugging startup issues.

### Health Check Failures

If the server reports unhealthy status:

1. **Check Postgres connectivity:**
   ```bash
   docker exec langgraph-postgres pg_isready -U postgres
   ```

2. **Check Redis connectivity:**
   ```bash
   docker exec langgraph-redis redis-cli ping
   ```

3. **View API logs:**
   ```bash
   docker logs langgraph-api
   ```

4. **Check health probe explicitly:**
   ```bash
   docker exec langgraph-api python /api/healthcheck.py
   ```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Port already in use | Another service on same port | Use `--port` to specify different port |
| `ImportError: no module named 'langraph_api'` | Missing `langgraph-cli[inmem]` extra | Install: `pip install 'langgraph-cli[inmem]'` |
| Container exits immediately | Missing dependencies or syntax error | Check logs: `docker logs langgraph-api` |
| `POSTGRES_URI` not working | Incorrect connection string | Use format: `postgresql://user:pass@host:port/db` |
| Volume permission errors | Docker user doesn't own volume | Run with `sudo` or adjust Docker group membership |

---

## Extending the CLI

### Custom Dockerfile Steps

Add build-time steps via config:

```json
{
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y graphviz",
    "ENV MY_CUSTOM_VAR=value",
    "COPY ./scripts /app/scripts"
  ]
}
```

These lines are inserted after the base image import but before dependency installation.

### Custom Build Commands

For JavaScript projects or specialized builds:

```bash
langgraph build -t my-graph \
  --install-command "npm ci --production" \
  --build-command "npm run build"
```

The CLI validates build commands to prevent shell injection (no pipes, semicolons, or backticks allowed).

**Evidence:** [`repo://libs/cli/langgraph_cli/config.py#L72-L78`](repo://libs/cli/langgraph_cli/config.py#L72-L78)

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy LangGraph

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install CLI
        run: pip install langgraph-cli
      
      - name: Build image
        run: langgraph build -t myregistry/my-graph:${{ github.sha }} --push
      
      - name: Deploy to LangSmith
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          langgraph deploy --name my-deployment \
            --git-url ${{ github.server_url }}/${{ github.repository }} \
            --git-commit ${{ github.sha }}
```

### Docker Registry Authentication

Log in before push:

```bash
docker login myregistry.azurecr.io
langgraph build -t myregistry.azurecr.io/my-graph:latest --push
```

Or use environment variables:

```bash
echo $REGISTRY_PASSWORD | docker login -u $REGISTRY_USER --password-stdin myregistry.azurecr.io
```

---

## Engine Runtime Modes

### Combined Queue-Worker (Default)

```bash
langgraph up --engine-runtime-mode combined_queue_worker
```

- Single container runs both task queue and worker threads
- Simplest setup, suitable for small to medium workloads
- Built-in autoscaling at the process level (adjustable via `N_JOBS_PER_WORKER`)

### Distributed Mode

```bash
langgraph up --engine-runtime-mode distributed
```

- **Orchestrator** container manages the task queue
- **Executor** containers (multiple instances) run graph invocations
- Horizontal scaling: add executor containers independently
- Better for high-concurrency production workloads

In distributed mode, the CLI launches separate base images and configures networking between components automatically.

**Evidence:** [`repo://libs/cli/langgraph_cli/cli.py#L172-L177`](repo://libs/cli/langgraph_cli/cli.py#L172-L177)
