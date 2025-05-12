---
search:
  boost: 2
---

# LangGraph CLI

**LangGraph CLI** is a multi-platform command-line tool for building and running the [LangGraph API server](./langgraph_server.md) locally. The resulting server includes all API endpoints for your graph's runs, threads, assistants, etc. as well as the other services required to run your agent, including a managed database for checkpointing and storage.

## Installation

The LangGraph CLI can be installed via pip:

=== "pip" 
    ```bash
    pip install langgraph-cli
    ```

## Commands

LangGraph CLI provides the following core functionality:

| Command | Description |
| -------- | -------|
| [`langgraph build`](../cloud/reference/cli.md#build) | Builds a Docker image for the [LangGraph API server](./langgraph_server.md) that can be directly deployed. |
| [`langgraph dev`](../cloud/reference/cli.md#dev) | Starts a lightweight development server that requires no Docker installation. This server is ideal for rapid development and testing. This is available in version 0.1.55 and up.
| [`langgraph dockerfile`](../cloud/reference/cli.md#dockerfile) | Generates a [Dockerfile](https://docs.docker.com/reference/dockerfile/) that can be used to build images for and deploy instances of the [LangGraph API server](./langgraph_server.md). This is useful if you want to further customize the dockerfile or deploy in a more custom way. |
| [`langgraph up`](../cloud/reference/cli.md#up) | Starts an instance of the [LangGraph API server](./langgraph_server.md) locally in a docker container. This requires the docker server to be running locally. It also requires a LangSmith API key for local development or a license key for production use. | 

For more information, see the [LangGraph CLI Reference](../cloud/reference/cli.md).
