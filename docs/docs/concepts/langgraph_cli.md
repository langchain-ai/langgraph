# LangGraph CLI

!!! info "Prerequisites"
    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)

The LangGraph CLI is a multi-platform command-line tool for building and running the [LangGraph API server](./langgraph_server.md) locally. The resulting server includes all API endpoints for your graph's runs, threads, assistants, etc. as well as the other services required to run your agent, including a managed database for checkpointing and storage.

## Installation

The LangGraph CLI can be installed via Homebrew (on macOS) or pip:

=== "Homebrew"
    ```bash
    brew install langgraph-cli
    ```

=== "pip" 
    ```bash
    pip install langgraph-cli
    ```

## Commands

The CLI provides the following core functionality:

### `build`

The `langgraph build` command builds a Docker image for the [LangGraph API server](./langgraph_server.md) that can be directly deployed.

### `dev`

!!! note "New in version 0.1.55"
    The `langgraph dev` command was introduced in langgraph-cli version 0.1.55.

!!! note "Python only"

    Currently, the CLI only supports Python >= 3.11.
    JS support is coming soon.

The `langgraph dev` command starts a lightweight development server that requires no Docker installation. This server is ideal for rapid development and testing, with features like:

- Hot reloading: Changes to your code are automatically detected and reloaded
- Debugger support: Attach your IDE's debugger for line-by-line debugging
- In-memory state with local persistence: Server state is stored in memory for speed but persisted locally between restarts

To use this command, you need to install the CLI with the "inmem" extra:

```bash
pip install -U "langgraph-cli[inmem]"
```

**Note**: This command is intended for local development and testing only. It is not recommended for production use. Since it does not use Docker, we recommend using virtual environments to manage your project's dependencies.

### `up`

The `langgraph up` command starts an instance of the [LangGraph API server](./langgraph_server.md) locally in a docker container. This requires thedocker server to be running locally. It also requires a LangSmith API key for local development or a license key for production use.

The server includes all API endpoints for your graph's runs, threads, assistants, etc. as well as the other services required to run your agent, including a managed database for checkpointing and storage.

### `dockerfile`

The `langgraph dockerfile` command generates a [Dockerfile](https://docs.docker.com/reference/dockerfile/) that can be used to build images for and deploy instances of the [LangGraph API server](./langgraph_server.md). This is useful if you want to further customize the dockerfile or deploy in a more custom way.

??? note "Updating your langgraph.json file"
    The `langgraph dockerfile` command translates all the configuration in your `langgraph.json` file into Dockerfile commands. When using this command, you will have to re-run it whenever you update your `langgraph.json` file. Otherwise, your changes will not be reflected when you build or run the dockerfile.

## Related

- [LangGraph CLI API Reference](../cloud/reference/cli.md)
