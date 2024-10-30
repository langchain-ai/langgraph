# LangGraph CLI

The LangGraph CLI is a multi-platform command-line tool for building and running the LangGraph API server locally. It offers an alternative to the [LangGraph Studio desktop app](./langgraph_studio.md) for developing and testing agents. The CLI is supported across all major OS's (Linux, Windows, MacOS).

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

### `up`

The `langgraph up` command starts an instance of the [LangGraph API server](./langgraph_server.md) locally. This requires docker to be installed and running locally. It also requires a LangSmith API key for local development or a license key for production use.

### `dockerfile`

The `langgraph dockerfile` command generates a [Dockerfile](https://docs.docker.com/reference/dockerfile/) that can be used to build images for and deploy instances of the [LangGraph API server](./langgraph_server.md). This is useful if you want to further customize the dockerfile or deploy in a more custom way.

