# LangGraph CLI

The LangGraph command line interface includes commands to build and run a LangGraph Platform API server locally in [Docker](https://www.docker.com/). For development and testing, you can use the CLI to deploy a local API server.

## Installation

1.  Ensure that Docker is installed (e.g. `docker --version`).
2.  Install the CLI package:

    === "Python"
        ```bash
        pip install langgraph-cli
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

The LangGraph CLI requires a JSON configuration file that follows this [schema](https://raw.githubusercontent.com/langchain-ai/langgraph/refs/heads/main/libs/cli/schemas/schema.json). It contains the following properties:

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        The LangGraph CLI defaults to using the configuration file <strong>langgraph.json</strong> in the current directory.
    </p>
</div>

=== "Python"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`dependencies`</span>     | **Required**. Array of dependencies for LangGraph Platform API server. Dependencies can be one of the following: <ul><li>A single period (`"."`), which will look for local Python packages.</li><li>The directory path where `pyproject.toml`, `setup.py` or `requirements.txt` is located.</br></br>For example, if `requirements.txt` is located in the root of the project directory, specify `"./"`. If it's located in a subdirectory called `local_package`, specify `"./local_package"`. Do not specify the string `"requirements.txt"` itself.</li><li>A Python package name.</li></ul> |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and returns an instance of `langgraph.graph.state.StateGraph` or `langgraph.graph.state.CompiledStateGraph`. See [how to rebuild a graph at runtime](../../cloud/deployment/graph_rebuild.md) for more details.</li></ul>                                    |
    | <span style="white-space: nowrap;">`auth`</span>             | _(Added in v0.0.11)_ Auth configuration containing the path to your authentication handler. Example: `./your_package/auth.py:auth`, where `auth` is an instance of `langgraph_sdk.Auth`. See [authentication guide](../../concepts/auth.md) for details.                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`base_image`</span>       | Optional. Base image to use for the LangGraph API server. Defaults to `langchain/langgraph-api` or `langchain/langgraphjs-api`. Use this to pin your builds to a particular version of the langgraph API, such as `"langchain/langgraph-server:0.2"`. See https://hub.docker.com/r/langchain/langgraph-server/tags for more details. (added in `langgraph-cli==0.2.8`) |
    | <span style="white-space: nowrap;">`image_distro`</span>     | Optional. Linux distribution for the base image. Must be either `"debian"` or `"wolfi"`. If omitted, defaults to `"debian"`. Available in `langgraph-cli>=0.2.11`.|
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search and/or time-to-live (TTL) to the BaseStore. Contains the following fields: <ul><li>`index` (optional): Configuration for semantic search indexing with fields `embed`, `dims`, and optional `fields`.</li><li>`ttl` (optional): Configuration for item expiration. An object with optional fields: `refresh_on_read` (boolean, defaults to `true`), `default_ttl` (float, lifespan in **minutes**, defaults to no expiration), and `sweep_interval_minutes` (integer, how often to check for expired items, defaults to no sweeping).</li></ul> |
    | <span style="white-space: nowrap;">`ui`</span>               | Optional. Named definitions of UI components emitted by the agent, each pointing to a JS/TS file. (added in `langgraph-cli==0.1.84`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    | <span style="white-space: nowrap;">`python_version`</span>   | `3.11`, `3.12`, or `3.13`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`pip_config_file`</span>  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
    | <span style="white-space: nowrap;">`pip_installer`</span> | _(Added in v0.3)_ Optional. Python package installer selector. It can be set to `"auto"`, `"pip"`, or `"uv"`. From version&nbsp;0.3 onward the default strategy is to run `uv pip`, which typically delivers faster builds while remaining a drop-in replacement. In the uncommon situation where `uv` cannot handle your dependency graph or the structure of your `pyproject.toml`, specify `"pip"` here to revert to the earlier behaviour. |
    | <span style="white-space: nowrap;">`keep_pkg_tools`</span> | _(Added in v0.3.4)_ Optional. Control whether to retain Python packaging tools (`pip`, `setuptools`, `wheel`) in the final image. Accepted values: <ul><li><code>true</code> : Keep all three tools (skip uninstall).</li><li><code>false</code> / omitted : Uninstall all three tools (default behaviour).</li><li><code>list[str]</code> : Names of tools <strong>to retain</strong>. Each value must be one of "pip", "setuptools", "wheel".</li></ul>. By default, all three tools are uninstalled. |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    | <span style="white-space: nowrap;">`checkpointer`</span>   | Configuration for the checkpointer. Contains a `ttl` field which is an object with the following keys: <ul><li>`strategy`: How to handle expired checkpoints (e.g., `"delete"`).</li><li>`sweep_interval_minutes`: How often to check for expired checkpoints (integer).</li><li>`default_ttl`: Default time-to-live for checkpoints in **minutes** (integer). Defines how long checkpoints are kept before the specified strategy is applied.</li></ul> |
    | <span style="white-space: nowrap;">`http`</span>            | HTTP server configuration with the following fields: <ul><li>`app`: Path to custom Starlette/FastAPI app (e.g., `"./src/agent/webapp.py:app"`). See [custom routes guide](../../how-tos/http/custom_routes.md).</li><li>`cors`: CORS configuration with fields for `allow_origins`, `allow_methods`, `allow_headers`, etc.</li><li>`configurable_headers`: Define which request headers to exclude or include as a run's configurable values.</li><li>`disable_assistants`: Disable `/assistants` routes</li><li>`disable_mcp`: Disable `/mcp` routes</li><li>`disable_meta`: Disable `/ok`, `/info`, `/metrics`, and `/docs` routes</li><li>`disable_runs`: Disable `/runs` routes</li><li>`disable_store`: Disable `/store` routes</li><li>`disable_threads`: Disable `/threads` routes</li><li>`disable_ui`: Disable `/ui` routes</li><li>`disable_webhooks`: Disable webhooks calls on run completion in all routes</li><li>`mount_prefix`: Prefix for mounted routes (e.g., "/my-deployment/api")</li></ul> |

=== "JS"

    | Key                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`graphs`</span>           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./src/graph.ts:variable`, where `variable` is an instance of `CompiledStateGraph`</li><li>`./src/graph.ts:makeGraph`, where `makeGraph` is a function that takes a config dictionary (`LangGraphRunnableConfig`) and returns an instance of `StateGraph` or `CompiledStateGraph`. See [how to rebuild a graph at runtime](../../cloud/deployment/graph_rebuild.md) for more details.</li></ul>                                    |
    | <span style="white-space: nowrap;">`env`</span>              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    | <span style="white-space: nowrap;">`store`</span>            | Configuration for adding semantic search and/or time-to-live (TTL) to the BaseStore. Contains the following fields: <ul><li>`index` (optional): Configuration for semantic search indexing with fields `embed`, `dims`, and optional `fields`.</li><li>`ttl` (optional): Configuration for item expiration. An object with optional fields: `refresh_on_read` (boolean, defaults to `true`), `default_ttl` (float, lifespan in **minutes**, defaults to no expiration), and `sweep_interval_minutes` (integer, how often to check for expired items, defaults to no sweeping).</li></ul> |
    | <span style="white-space: nowrap;">`node_version`</span>     | Specify `node_version: 20` to use LangGraph.js.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | <span style="white-space: nowrap;">`dockerfile_lines`</span> | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    | <span style="white-space: nowrap;">`checkpointer`</span>   | Configuration for the checkpointer. Contains a `ttl` field which is an object with the following keys: <ul><li>`strategy`: How to handle expired checkpoints (e.g., `"delete"`).</li><li>`sweep_interval_minutes`: How often to check for expired checkpoints (integer).</li><li>`default_ttl`: Default time-to-live for checkpoints in **minutes** (integer). Defines how long checkpoints are kept before the specified strategy is applied.</li></ul> |

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

    #### Using Wolfi Base Images

    You can specify the Linux distribution for your base image using the `image_distro` field. Valid options are `debian` or `wolfi`. Wolfi is the recommended option as it provides smaller and more secure images. This is available in `langgraph-cli>=0.2.11`.

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "image_distro": "wolfi"
    }
    ```

    #### Adding semantic search to the store

    All deployments come with a DB-backed BaseStore. Adding an "index" configuration to your `langgraph.json` will enable [semantic search](../deployment/semantic_search.md) within the BaseStore of your deployment.

    The `index.fields` configuration determines which parts of your documents to embed:

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

    #### Configuring Store Item Time-to-Live (TTL)

    You can configure default data expiration for items/memories in the BaseStore using the `store.ttl` key. This determines how long items are retained after they are last accessed (with reads potentially refreshing the timer based on `refresh_on_read`). Note that these defaults can be overwritten on a per-call basis by modifying the corresponding arguments in `get`, `search`, etc.
    
    The `ttl` configuration is an object containing optional fields:

    - `refresh_on_read`: If `true` (the default), accessing an item via `get` or `search` resets its expiration timer. Set to `false` to only refresh TTL on writes (`put`).
    - `default_ttl`: The default lifespan of an item in **minutes**. If not set, items do not expire by default.
    - `sweep_interval_minutes`: How frequently (in minutes) the system should run a background process to delete expired items. If not set, sweeping does not occur automatically.

    Here is an example enabling a 7-day TTL (10080 minutes), refreshing on reads, and sweeping every hour:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "memory_agent": "./agent/graph.py:graph"
      },
      "store": {
        "ttl": {
          "refresh_on_read": true,
          "sweep_interval_minutes": 60,
          "default_ttl": 10080 
        }
      }
    }
    ```

    #### Configuring Checkpoint Time-to-Live (TTL)

    You can configure the time-to-live (TTL) for checkpoints using the `checkpointer` key. This determines how long checkpoint data is retained before being automatically handled according to the specified strategy (e.g., deletion). The `ttl` configuration is an object containing:

    - `strategy`: The action to take on expired checkpoints (currently `"delete"` is the only accepted option).
    - `sweep_interval_minutes`: How frequently (in minutes) the system checks for expired checkpoints.
    - `default_ttl`: The default lifespan of a checkpoint in **minutes**.

    Here's an example setting a default TTL of 30 days (43200 minutes):

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "chat": "./chat/graph.py:graph"
      },
      "checkpointer": {
        "ttl": {
          "strategy": "delete",
          "sweep_interval_minutes": 10,
          "default_ttl": 43200
        }
      }
    }
    ```

    In this example, checkpoints older than 30 days will be deleted, and the check runs every 10 minutes.


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
    | `--wait-for-client`           | `False`          | Wait for a debugger client to connect to the debug port before starting the server   |
    | `--no-browser`                |                  | Skip automatically opening the browser when the server starts                       |
    | `--studio-url TEXT`           |                  | URL of the LangGraph Studio instance to connect to. Defaults to https://smith.langchain.com |
    | `--allow-blocking`            | `False`          | Do not raise errors for synchronous I/O blocking operations in your code (added in `0.2.6`)           |
    | `--tunnel`                    | `False`          | Expose the local server via a public tunnel (Cloudflare) for remote frontend access. This avoids issues with browsers like Safari or networks blocking localhost connections        |
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
    | `--wait-for-client`           | `False`          | Wait for a debugger client to connect to the debug port before starting the server   |
    | `--no-browser`                |                  | Skip automatically opening the browser when the server starts                       |
    | `--studio-url TEXT`           |                  | URL of the LangGraph Studio instance to connect to. Defaults to https://smith.langchain.com |
    | `--allow-blocking`            | `False`          | Do not raise errors for synchronous I/O blocking operations in your code            |
    | `--tunnel`                    | `False`          | Expose the local server via a public tunnel (Cloudflare) for remote frontend access. This avoids issues with browsers or networks blocking localhost connections        |
    | `--help`                      |                  | Display command documentation                                                       |

### `build`

=== "Python"

    Build LangGraph Platform API server Docker image.

    **Usage**

    ```
    langgraph build [OPTIONS]
    ```

    **Options**

    | Option               | Default          | Description                                                                                                     |
    | -------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
    | `--pull / --no-pull` | `--pull`         | Build with latest remote Docker image. Use `--no-pull` for running the LangGraph Platform API server with locally built images. |
    | `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
    | `--help`             |                  | Display command documentation.                                                                                               |

=== "JS"

    Build LangGraph Platform API server Docker image.

    **Usage**

    ```
    npx @langchain/langgraph-cli build [OPTIONS]
    ```

    **Options**

    | Option               | Default          | Description                                                                                                     |
    | -------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
    | `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
    | `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
    | `--no-pull`          |                  | Use locally built images. Defaults to `false` to build with latest remote Docker image.                                      |
    | `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
    | `--help`             |                  | Display command documentation.                                                                                               |


### `up`

=== "Python"

    Start LangGraph API server. For local testing, requires a LangSmith API key with access to LangGraph Platform. Requires a license key for production use.

    **Usage**

    ```
    langgraph up [OPTIONS]
    ```

    **Options**

    | Option                       | Default                   | Description                                                                                                             |
    | ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | `--wait`                     |                           | Wait for services to start before returning. Implies --detach                                                           |
    | `--base-image TEXT`          | `langchain/langgraph-api`  | Base image to use for the LangGraph API server. Pin to specific versions using version tags.                            |
    | `--image TEXT`               |                           | Docker image to use for the langgraph-api service. If specified, skips building and uses this image directly.           |
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

    Start LangGraph API server. For local testing, requires a LangSmith API key with access to LangGraph Platform. Requires a license key for production use.

    **Usage**

    ```
    npx @langchain/langgraph-cli up [OPTIONS]
    ```

    **Options**

    | Option                                                                 | Default                   | Description                                                                                                             |
    | ---------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
    | <span style="white-space: nowrap;">`--wait`</span>                     |                           | Wait for services to start before returning. Implies --detach                                                           |
    | <span style="white-space: nowrap;">`--base-image TEXT`</span>          | <span style="white-space: nowrap;">`langchain/langgraph-api`</span> | Base image to use for the LangGraph API server. Pin to specific versions using version tags. |
    | <span style="white-space: nowrap;">`--image TEXT`</span>               |                           | Docker image to use for the langgraph-api service. If specified, skips building and uses this image directly. |
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

    Generate a Dockerfile for building a LangGraph Platform API server Docker image.

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

    ADD ./graphs /deps/outer-graphs/src
    RUN set -ex && \
        for line in '[project]' \
                    'name = "graphs"' \
                    'version = "0.1"' \
                    '[tool.setuptools.package-data]' \
                    '"*" = ["**/*"]'; do \
            echo "$line" >> /deps/outer-graphs/pyproject.toml; \
        done

    RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

    ENV LANGSERVE_GRAPHS='{"agent": "/deps/outer-graphs/src/agent.py:graph", "storm": "/deps/outer-graphs/src/storm.py:graph"}'
    ```

    ???+ note "Updating your langgraph.json file"
         The `langgraph dockerfile` command translates all the configuration in your `langgraph.json` file into Dockerfile commands. When using this command, you will have to re-run it whenever you update your `langgraph.json` file. Otherwise, your changes will not be reflected when you build or run the dockerfile.

=== "JS"

    Generate a Dockerfile for building a LangGraph Platform API server Docker image.

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
