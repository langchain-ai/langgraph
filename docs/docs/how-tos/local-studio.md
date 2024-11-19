# How to connect a local agent to LangGraph Studio

When developing your agent, you may want to connect it to [LangGraph Studio](../concepts/langgraph_studio.md).
This can be useful to visualize your graph as you work on it, interact with it, and debug any edge cases.

There are two ways to do this:

- [LangGraph Desktop](../concepts/langgraph_studio.md#desktop-app): Application, Mac only, requires Docker
- [Development Server](../concepts/langgraph_studio.md#dev-server): Python package, all platforms, no Docker

In this guide we will cover how to use the development server as that is generally an easier and better experience.

## Setup your application

First, you will need to setup your application in the proper format.
This means defining a `langgraph.json` file which contains paths to your agent(s).
See [this guide](../concepts/application_structure.md) for information on how to do so.

## Install langgraph-cli

You will need to install [`langgraph-cli`](../cloud/reference/cli.md#langgraph-cli) (version `0.1.55` or higher).
You will need to make sure to install the `inmem` extras.

```shell
pip install "langgraph-cli[inmem]==0.1.55"
```

## Run the development server

Make sure you are in the directory with your `langgraph.json` and then start the development server.

```shell
langgraph dev
```

This will look for the `langgraph.json` file in your current directory. 
In there, it will find the paths to the graph(s), and start those up.
It will then automatically connect to the cloud-hosted studio.

## Use the studio

After connecting to the studio, a browser window should automatically pop up.
This will use the cloud hosted studio UI to connect to your local development server.
Your graph is still running locally, the UI is connecting to visualizing the agent and threads that are defined locally.

The graph will always use the most up-to-date code, so you will be able to change the underlying code and have it automatically reflected in the studio.
This is useful for debugging workflows.
You can run your graph in the UI until it messes up, go in and change your code, and then rerun from the node that failed.