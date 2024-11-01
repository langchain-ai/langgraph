# LangGraph CLI

The LangGraph command line interface includes commands to build and run a LangGraph Cloud API server locally in [Docker](https://www.docker.com/). For development and testing, you can use the CLI to deploy a local API server as an alternative to the [Studio desktop app](../../concepts/langgraph_studio.md).

## Installation

1. Ensure that Docker is installed (e.g. `docker --version`).
2. Install the `langgraph-cli` package:
 
    === "pip"
        ```bash    
        pip install langgraph-cli
        ```

    === "Homebrew (MacOS only)"
        ```bash
        brew install langgraph-cli
        ```
 
3. Run the command `langgraph --help` to confirm that the CLI is installed.

[](){#langgraph.json}

## Configuration File

The LangGraph CLI requires a JSON configuration file with the following keys:

| Key                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dependencies`     | **Required**. Array of dependencies for LangGraph Cloud API server. Dependencies can be one of the following: (1) `"."`, which will look for local Python packages, (2) `pyproject.toml`, `setup.py` or `requirements.txt` in the app directory `"./local_package"`, or (3) a package name.                                                                                                                                                                                                                                                  |
| `graphs`           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.</li></ul> |
| `env`              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `python_version`   | `3.11` or `3.12`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `pip_config_file`  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dockerfile_lines` | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        The LangGraph CLI defaults to using the configuration file <strong>langgraph.json</strong> in the current directory.
    </p>
</div>

Example:

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_graph_id": "./your_package/your_file.py:variable"
  },
  "env": "./.env"
}
```

Example:

```json
{
  "python_version": "3.11",
  "dependencies": ["langchain_openai", "."],
  "graphs": {
    "my_graph_id": "./your_package/your_file.py:make_graph"
  },
  "env": {
    "OPENAI_API_KEY": "secret-key"
  }
}
```

## Commands

The base command for the LangGraph CLI is `langgraph`.

**Usage**

```
langgraph [OPTIONS] COMMAND [ARGS]
```

### `build`

Build LangGraph Cloud API server Docker image.

**Usage**

```
langgraph build [OPTIONS]
```

**Options**

| Option               | Default          | Description                                                                                                                  |
|----------------------|------------------|------------------------------------------------------------------------------------------------------------------------------|
| `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
| `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
| `--pull / --no-pull` | `--pull`         | Build with latest remote Docker image. Use `--no-pull` for running the LangGraph Cloud API server with locally built images. |
| `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
| `--help`             |                  | Display command documentation.                                                                                               |

### `up`

Start langgraph API server. For local testing, requires a LangSmith API key with access to LangGraph Cloud closed beta. Requires a license key for production use.

**Usage**

```
langgraph up [OPTIONS]
```

**Options**

| Option                       | Default                   | Description                                                                                                           |
|------------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `--wait`                     |                           | Wait for services to start before returning. Implies --detach                                                         |
| `--postgres-uri TEXT`        | Local database            | Postgres URI to use for the database.                                                                                 |
| `--watch`                    |                           | Restart on file changes                                                                                               |
| `--debugger-base-url TEXT`   | `http://127.0.0.1:[PORT]` | URL used by the debugger to access LangGraph API.                                                                     |
| `--debugger-port INTEGER`    |                           | Pull the debugger image locally and serve the UI on specified port                                                    |
| `--verbose`                  |                           | Show more output from the server logs.                                                                                |
| `-c, --config FILE`          | `langgraph.json`          | Path to configuration file declaring dependencies, graphs and environment variables.                                  |
| `-d, --docker-compose FILE`  |                           | Path to docker-compose.yml file with additional services to launch.                                                   |
| `-p, --port INTEGER`         | `8123`                    | Port to expose. Example: `langgraph test --port 8000`                                                                 |
| `--pull / --no-pull`         | `pull`                    | Pull latest images. Use --no-pull for running the server with locally-built images. Example: `langgraph up --no-pull` |
| `--recreate / --no-recreate` | `no-recreate`             | Recreate containers even if their configuration and image haven't changed                                             |
| `--help`                     |                           | Display command documentation.                                                                                        |

### `dockerfile`

Generate a Dockerfile for building a LangGraph Cloud API server Docker image.

**Usage**

```
langgraph dockerfile [OPTIONS] SAVE_PATH
```

**Options**

| Option              | Default          | Description                                                                                                     |
|---------------------|------------------|-----------------------------------------------------------------------------------------------------------------|
| `-c, --config FILE` | `langgraph.json` | Path to the [configuration file](#configuration-file) declaring dependencies, graphs and environment variables. |
| `--help`            |                  | Show this message and exit.                                                                                     |

Example:

```bash
langgraph dockerfile -c langgraph.json Dockerfile
```

Would generate something like the following:

```text
FROM langchain/langgraph-api:3.11

ADD ./pipconf.txt /pipconfig.txt

RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain_community langchain_anthropic langchain_openai wikipedia scikit-learn

ADD ./graphs /deps/__outer_graphs/src
RUN set -ex && \
    for line in '[project]' \
                'name = "graphs"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \
    done

RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_graphs/src/agent.py:graph", "storm": "/deps/__outer_graphs/src/storm.py:graph"}'
```

You can then customize, build images, push, and deploy from this file.
