# Application Structure

!!! info "Prerequisites"

    - [LangGraph Server](./langgraph_server.md)
    - [LangGraph Glossary](./low_level.md)

## Overview

A LangGraph application consists of one or more graphs, a LangGraph API Configuration file (`langgraph.json`), a file that specifies dependencies, and an optional .env file that specifies environment variables.

This guide shows a typical structure for a LangGraph application and shows how the required information to deploy a LangGraph application using the LangGraph Platform is specified.

## Key Concepts

To deploy using the LangGraph Platform, the following information should be provided:

1. A [LangGraph API Configuration file](#configuration-file) (`langgraph.json`) that specifies the dependencies, graphs, environment variables to use for the application.
2. The [graphs](#graphs) that implement the logic of the application.
3. A file that specifies [dependencies](#dependencies) required to run the application.
4. [Environment variable](#environment-variables) that are required for the application to run.

## File Structure

Below are examples of directory structures for Python and JavaScript applications:

=== "Python (requirements.txt)"

    ```plaintext
    my-app/
    ├── my_agent # all project code lies within here
    │   ├── utils # utilities for your graph
    │   │   ├── __init__.py
    │   │   ├── tools.py # tools for your graph
    │   │   ├── nodes.py # node functions for you graph
    │   │   └── state.py # state definition of your graph
    │   ├── __init__.py
    │   └── agent.py # code for constructing your graph
    ├── .env # environment variables
    ├── requirements.txt # package dependencies
    └── langgraph.json # configuration file for LangGraph
    ```
=== "Python (pyproject.toml)"

    ```plaintext
    my-app/
    ├── my_agent # all project code lies within here
    │   ├── utils # utilities for your graph
    │   │   ├── __init__.py
    │   │   ├── tools.py # tools for your graph
    │   │   ├── nodes.py # node functions for you graph
    │   │   └── state.py # state definition of your graph
    │   ├── __init__.py
    │   └── agent.py # code for constructing your graph
    ├── .env # environment variables
    ├── langgraph.json  # configuration file for LangGraph
    └── pyproject.toml # dependencies for your project
    ```

=== "JS (package.json)"

    ```plaintext
    my-app/
    ├── src # all project code lies within here
    │   ├── utils # optional utilities for your graph
    │   │   ├── tools.ts # tools for your graph
    │   │   ├── nodes.ts # node functions for you graph
    │   │   └── state.ts # state definition of your graph
    │   └── agent.ts # code for constructing your graph
    ├── package.json # package dependencies
    ├── .env # environment variables
    └── langgraph.json # configuration file for LangGraph
    ```

!!! note

    The directory structure of a LangGraph application can vary depending on the programming language and the package manager used.


## Configuration File

The `langgraph.json` file is a JSON file that specifies the dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application.

The file supports specification of the following information:


| Key                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dependencies`     | **Required**. Array of dependencies for LangGraph API server. Dependencies can be one of the following: (1) `"."`, which will look for local Python packages, (2) `pyproject.toml`, `setup.py` or `requirements.txt` in the app directory `"./local_package"`, or (3) a package name.                                                                                                                                                                                                                                                        |
| `graphs`           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.</li></ul> |
| `env`              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `python_version`   | `3.11` or `3.12`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `pip_config_file`  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dockerfile_lines` | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
!!! tip

    The LangGraph CLI defaults to using the configuration file **langgraph.json** in the current directory.


### Examples

=== "Python"

    * The dependencies involve a custom local package and the `langchain_openai` package.
    * A single graph will be loaded from the file `./your_package/your_file.py` with the variable `variable`.
    * The environment variables are loaded from the `.env` file.

    ```json
    {
        "dependencies": [
            "langchain_openai",
            "./your_package"
        ],
        "graphs": {
            "my_agent": "./your_package/your_file.py:agent"
        },
        "env": "./.env"
    }
    ```

=== "JavaScript"

    * The dependencies will be loaded from a dependency file in the local directory (e.g., `package.json`).
    * A single graph will be loaded from the file `./your_package/your_file.js` with the function `agent`.
    * The environment variable `OPENAI_API_KEY` is set inline.

    ```json
    {
        "dependencies": [
            "."
        ],
        "graphs": {
            "my_agent": "./your_package/your_file.js:agent"
        },
        "env": {
            "OPENAI_API_KEY": "secret-key"
        }
    }
    ```

## Dependencies

A LangGraph application may depend on other Python packages or JavaScript libraries (depending on the programming language in which the application is written).

You will generally need to specify the following information for dependencies to be set up correctly:

1. A file in the directory that specifies the dependencies (e.g., `requirements.txt`, `pyproject.toml`, or `package.json`).
2. A `dependencies` key in the [LangGraph configuration file](#configuration-file) that specifies the dependencies required to run the LangGraph application.
3. Any additional binaries or system libraries can be specified using `dockerfile_lines` key in the [LangGraph configuration file](#configuration-file).

## Graphs

Use the `graphs` key in the [LangGraph configuration file](#configuration-file) to specify which graphs will be available in the deployed LangGraph application.

You can specify one or more graphs in the configuration file. Each graph is identified by a name (which should be unique) and a path for either: (1) the compiled graph or (2) a function that makes a graph is defined.

## Environment Variables

If you're working with a deployed LangGraph application locally, you can configure environment variables in the `env` key of the [LangGraph configuration file](#configuration-file).

For a production deployment, you will typically want to configure the environment variables in the deployment environment.

## Related

Please see the following resources for more information:

- How-to guides for [Application Structure](../how-tos/index.md#application-structure).
