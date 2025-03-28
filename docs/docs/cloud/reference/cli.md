# LangGraph CLI

The LangGraph command line interface includes commands to build and run a LangGraph Cloud API server locally in [Docker](https://www.docker.com/). For development and testing, you can use the CLI to deploy a local API server.

## Installation

1.  Ensure that Docker is installed (e.g. `docker --version`).
2.  Install the CLI package:

    === "Python"
        ```bash
        pip install langgraph-cli

        # Install via Homebrew
        brew install langgraph-cli
        ```

    === "JS"
        ```bash
        npx @langchain/langgraph-cli

        # Install globally, will be available as `langgraphjs`
        npm install -g @langchain/langgraph-cli
        ```

3.  Run the command `langgraph --help` or `npx @langchain/langgraph-cli --help` to confirm that the CLI is working correctly.

[](){#langgraph.json}

## Configuration File {#configuration-file}

The LangGraph CLI requires a JSON configuration file with the following keys:

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        The LangGraph CLI defaults to using the configuration file <strong>langgraph.json</strong> in the current directory.
    </p>
</div>

=== "Python"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`dependencies`</span>     | **Required**. Array of dependencies for LangGraph Cloud API server. Dependencies can be one of the following: <ul><li>A single period (`"."`), which will look for local Python packages.</li><li>The directory path where `pyproject.toml`, `setup.py` or `requirements.txt` is located.</br></br>For example, if `requirements.txt` is located in the root of the project directory, specify `"./"`. If it's located in a subdirectory called `local_package`, specify `"./local_package"`. Do not specify the string `"requirements.txt"` itself.</li><li>A Python package name.</li></ul> |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.</li></ul>                                    |
    | <span style="white-space: nowrap;">`auth`</span>             | _(Added in v0.0.11)_ Auth configuration containing the path to your authentication handler. Example: `./your_package/auth.py:auth`, where `auth` is an instance of `langgraph_sdk.Auth`. See [authentication guide](../../concepts/auth.md) for details.                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search to the BaseStore. Contains the following fields: <ul><li>`index`: Configuration for semantic search indexing with fields:<ul><li>`embed`: Embedding provider (e.g., "openai:text-embedding-3-small") or path to custom embedding function</li><li>`dims`: Dimension size of the embedding model. Used to initialize the vector table.</li><li>`fields` (optional): List of fields to index. Defaults to `["$"]`, which means to index entire documents. Can be specific fields like `["text", "summary", "some.value"]`</li></ul></li></ul> |
    | <span style="white-space: nowrap;">`python_version`</span>   | `3.11`, `3.12`, or `3.13`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`pip_config_file`</span>  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    | <span style="white-space: nowrap;">`http`</span>            | HTTP server configuration with the following fields: <ul><li>`app`: Path to custom Starlette/FastAPI app (e.g., `"./src/agent/webapp.py:app"`). See [custom routes guide](../../how-tos/http/custom_routes.md).</li><li>`disable_assistants`: Disable `/assistants` routes</li><li>`disable_threads`: Disable `/threads` routes</li><li>`disable_runs`: Disable `/runs` routes</li><li>`disable_store`: Disable `/store` routes</li><li>`disable_meta`: Disable `/ok`, `/info`, `/metrics`, and `/docs` routes</li><li>`cors`: CORS configuration with fields for `allow_origins`, `allow_methods`, `allow_headers`, etc.</li></ul> |

=== "JS"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./src/graph.ts:variable`, where `variable` is an instance of `CompiledStateGraph`</li><li>`./src/graph.ts:makeGraph`, where `makeGraph` is a function that takes a config dictionary (`LangGraphRunnableConfig`) and creates an instance of `StateGraph` / `CompiledStateGraph`.</li></ul>                                    |
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search to the BaseStore. Contains the following fields: <ul><li>`index`: Configuration for semantic search indexing with fields:<ul><li>`embed`: Embedding provider (e.g., "openai:text-embedding-3-small") or path to custom embedding function</li><li>`dims`: Dimension size of the embedding model. Used to initialize the vector table.</li><li>`fields` (optional): List of fields to index. Defaults to `["$"]`, which means to index entire documents. Can be specific fields like `["text", "summary", "some.value"]`</li></ul></li></ul> |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

### Examples

=== "Python"
    
    #### Basic Configuration

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      }
    }
    ```

    #### Adding semantic search to the store

    All deployments come with a DB-backed BaseStore. Adding an "index" configuration to your `langgraph.json` will enable [semantic search](../deployment/semantic_search.md) within the BaseStore of your deployment.

    The `fields` configuration determines which parts of your documents to embed:

    - If omitted or set to `["$"]`, the entire document will be embedded
    - To embed specific fields, use JSON path notation: `["metadata.title", "content.text"]`
    - Documents missing specified fields will still be stored but won't have embeddings for those fields
    - You can still override which fields to embed on a specific item at `put` time using the `index` parameter

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "index": {
          "embed": "openai:text-embedding-3-small",
          "dims": 1536,
          "fields": ["$"]
        }
      }
    }
    ```

    !!! note "Common model dimensions" 
        - `openai:text-embedding-3-large`: 3072 
        - `openai:text-embedding-3-small`: 1536 
        - `openai:text-embedding-ada-002`: 1536 
        - `cohere:embed-english-v3.0`: 1024 
        - `cohere:embed-english-light-v3.0`: 384 
        - `cohere:embed-multilingual-v3.0`: 1024 
        - `cohere:embed-multilingual-light-v3.0`: 384

    #### Semantic search with a custom embedding function

    If you want to use semantic search with a custom embedding function, you can pass a path to a custom embedding function:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "index": {
          "embed": "./embeddings.py:embed_texts",
          "dims": 768,
          "fields": ["text", "summary"]
        }
      }
    }
    ```

    The `embed` field in store configuration can reference a custom function that takes a list of strings and returns a list of embeddings. Example implementation:

    ```python
    # embeddings.py
    def embed_texts(texts: list[str]) -> list[list[float]]:
        """Custom embedding function for semantic search."""
        # Implementation using your preferred embedding model
        return [[0.1, 0.2, ...] for _ in texts]  # dims-dimensional vectors
    ```

    #### Adding custom authentication

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "auth": {
        "path": "./auth.py:auth",
        "openapi": {
          "securitySchemes": {
            "apiKeyAuth": {
              "type": "apiKey",
              "in": "header",
              "name": "X-API-Key"
            }
          },
          "security": [{ "apiKeyAuth": [] }]
        },
        "disable_studio_auth": false
      }
    }
    ```

    See the [authentication conceptual guide](../../concepts/auth.md) for details, and the [setting up custom authentication](../../tutorials/auth/getting_started.md) guide for a practical walk through of the process.


=== "JS"
    
    #### Basic Configuration

    ```json
    {
      "graphs": {
        "chat": "./src/graph.ts:graph"
      }
    }
    ```


## Commands

**Usage**

=== "Python"

    The base command for the LangGraph CLI is `langgraph`.

    ```
    langgraph [OPTIONS] COMMAND [ARGS]
    ```
=== "JS"

    The base command for the LangGraph.js CLI is `langgraphjs`. 

    ```
    npx @langchain/langgraph-cli [OPTIONS] COMMAND [ARGS]
    ```

    We recommend using `npx` to always use the latest version of the CLI.

### `dev`

=== "Python"

    Run LangGraph API server in development mode with hot reloading and debugging capabilities. This lightweight server requires no Docker installation and is suitable for development and testing. State is persisted to a local directory.

    !!! note

        Currently, the CLI only supports Python >= 3.11.

    **Installation**

    This command requires the "inmem" extra to be installed:

    ```bash
    pip install -U "langgraph-cli[inmem]"
    ```

    **Usage**

    ```
    langgraph dev [OPTIONS]
    ```

    **Options**

    | Option                        | Default          | Description                                                                         |
    | ----------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
    | `-c, --config FILE`           | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables |
    | `--host TEXT`                 | `127.0.0.1`      | Host to bind the server to                                                          |
    | `--port INTEGER`              | `2024`           | Port to bind the server to                                                          |
    | `--no-reload`                 |                  | Disable auto-reload                                                                 |
    | `--n-jobs-per-worker INTEGER` |                  | Number of jobs per worker. Default is 10                                            |
    | `--debug-port INTEGER`        |                  | Port for debugger to listen on                                                      |
    | `--help`                      |                  | Display command documentation                                                       |


=== "JS"

    Run LangGraph API server in development mode with hot reloading capabilities. This lightweight server requires no Docker installation and is suitable for development and testing. State is persisted to a local directory.

    **Usage**

    ```
    npx @langchain/langgraph-cli dev [OPTIONS]
    ```

    **Options**

    | Option                        | Default          | Description                                                                         |
    | ----------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
    | `-c, --config FILE`           | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables |
    | `--host TEXT`                 | `127.0.0.1`      | Host to bind the server to                                                          |
    | `--port INTEGER`              | `2024`           | Port to bind the server to                                                          |
    | `--no-reload`                 |                  | Disable auto-reload                                                                 |
    | `--n-jobs-per-worker INTEGER` |                  | Number of jobs per worker. Default is 10                                            |
    | `--debug-port INTEGER`        |                  | Port for debugger to listen on                                                      |
    | `--help`                      |                  | Display command documentation                                                       |

### `build`

=== "Python"

    Build LangGraph Cloud API server Docker image.

    **Usage**

    ```
    langgraph build [OPTIONS]
    ```

    **Options**

    | Option               | Default          | Description                                                                                                                  |
    | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
    | `--pull / --no-pull` | `--pull`         | Build with latest remote Docker image. Use `--no-pull` for running the LangGraph Cloud API server with locally built images. |
    | `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
    | `--help`             |                  | Display command documentation.                                                                                               |

=== "JS"

    Build LangGraph Cloud API server Docker image.

    **Usage**

    ```
    npx @langchain/langgraph-cli build [OPTIONS]
    ```

    **Options**

    | Option               | Default          | Description                                                                                                                  |
    | -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
    | `--no-pull`          |                  | Use locally built images. Defaults to `false` to build with latest remote Docker image.                                      |
    | `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
    | `--help`             |                  | Display command documentation.                                                                                               |


### `up`

=== "Python"

    Start LangGraph API server. For local testing, requires a LangSmith API key with access to LangGraph Cloud closed beta. Requires a license key for production use.

    **Usage**

    ```
    langgraph up [OPTIONS]
    ```

    **Options**

    | Option                       | Default                   | Description                                                                                                             |
    | ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | `--wait`                     |                           | Wait for services to start before returning. Implies --detach                                                           |
    | `--postgres-uri TEXT`        | Local database            | Postgres URI to use for the database.                                                                                   |
    | `--watch`                    |                           | Restart on file changes                                                                                                 |
    | `--debugger-base-url TEXT`   | `http://127.0.0.1:[PORT]` | URL used by the debugger to access LangGraph API.                                                                       |
    | `--debugger-port INTEGER`    |                           | Pull the debugger image locally and serve the UI on specified port                                                      |
    | `--verbose`                  |                           | Show more output from the server logs.                                                                                  |
    | `-c, --config FILE`          | `langgraph.json`          | Path to configuration file declaring dependencies, graphs and environment variables.                                    |
    | `-d, --docker-compose FILE`  |                           | Path to docker-compose.yml file with additional services to launch.                                                     |
    | `-p, --port INTEGER`         | `8123`                    | Port to expose. Example: `langgraph up --port 8000`                                                                     |
    | `--pull / --no-pull`         | `pull`                    | Pull latest images. Use `--no-pull` for running the server with locally-built images. Example: `langgraph up --no-pull` |
    | `--recreate / --no-recreate` | `no-recreate`             | Recreate containers even if their configuration and image haven't changed                                               |
    | `--help`                     |                           | Display command documentation.                                                                                          |

=== "JS"

    Start LangGraph API server. For local testing, requires a LangSmith API key with access to LangGraph Cloud closed beta. Requires a license key for production use.

    **Usage**

    ```
    npx @langchain/langgraph-cli up [OPTIONS]
    ```

    **Options**

    | Option                                                                 | Default                   | Description                                                                                                             |
    | ---------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`--wait`</span>                     |                           | Wait for services to start before returning. Implies --detach                                                           |
    | <span style="white-space: nowrap;">`--postgres-uri TEXT`</span>        | Local database            | Postgres URI to use for the database.                                                                                   |
    | <span style="white-space: nowrap;">`--watch`</span>                    |                           | Restart on file changes                                                                                                 |
    | <span style="white-space: nowrap;">`-c, --config FILE`</span>          | `langgraph.json`          | Path to configuration file declaring dependencies, graphs and environment variables.                                    |
    | <span style="white-space: nowrap;">`-d, --docker-compose FILE`</span>  |                           | Path to docker-compose.yml file with additional services to launch.                                                     |
    | <span style="white-space: nowrap;">`-p, --port INTEGER`</span>         | `8123`                    | Port to expose. Example: `langgraph up --port 8000`                                                                     |
    | <span style="white-space: nowrap;">`--no-pull`</span>                  |                           | Use locally built images. Defaults to `false` to build with latest remote Docker image.                                 |
    | <span style="white-space: nowrap;">`--recreate`</span>                 |                           | Recreate containers even if their configuration and image haven't changed                                               |
    | <span style="white-space: nowrap;">`--help`</span>                     |                           | Display command documentation.                                                                                          |

### `dockerfile`

=== "Python"

    Generate a Dockerfile for building a LangGraph Cloud API server Docker image.

    **Usage**

    ```
    langgraph dockerfile [OPTIONS] SAVE_PATH
    ```

    **Options**

    | Option              | Default          | Description                                                                                                     |
    | ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `-c, --config FILE` | `langgraph.json` | Path to the [configuration file](#configuration-file) declaring dependencies, graphs and environment variables. |
    | `--help`            |                  | Show this message and exit.                                                                                     |

    Example:

    ```bash
    langgraph dockerfile -c langgraph.json Dockerfile
    ```

    This generates a Dockerfile that looks similar to:

    ```dockerfile
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

    ???+ note "Updating your langgraph.json file"
         The `langgraph dockerfile` command translates all the configuration in your `langgraph.json` file into Dockerfile commands. When using this command, you will have to re-run it whenever you update your `langgraph.json` file. Otherwise, your changes will not be reflected when you build or run the dockerfile.

=== "JS"

    Generate a Dockerfile for building a LangGraph Cloud API server Docker image.

    **Usage**

    ```
    npx @langchain/langgraph-cli dockerfile [OPTIONS] SAVE_PATH
    ```

    **Options**

    | Option              | Default          | Description                                                                                                     |
    | ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `-c, --config FILE` | `langgraph.json` | Path to the [configuration file](#configuration-file) declaring dependencies, graphs and environment variables. |
    | `--help`            |                  | Show this message and exit.                                                                                     |

    Example:

    ```bash
    npx @langchain/langgraph-cli dockerfile -c langgraph.json Dockerfile
    ```

    This generates a Dockerfile that looks similar to:

    ```dockerfile
    FROM langchain/langgraphjs-api:20
    
    ADD . /deps/agent
    
    RUN cd /deps/agent && yarn install
    
    ENV LANGSERVE_GRAPHS='{"agent":"./src/react_agent/graph.ts:graph"}'
    
    WORKDIR /deps/agent
    
    RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts
    ```

    ???+ note "Updating your langgraph.json file"
         The `npx @langchain/langgraph-cli dockerfile` command translates all the configuration in your `langgraph.json` file into Dockerfile commands. When using this command, you will have to re-run it whenever you update your `langgraph.json` file. Otherwise, your changes will not be reflected when you build or run the dockerfile.
