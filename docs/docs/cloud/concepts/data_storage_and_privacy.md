# Data Storage and Privacy

This document describes how data is processed in the LangGraph CLI and the LangGraph Server for both the in-memory server (`langgraph dev`) and the local docker server (`langgraph up`). It also describes what data is tracked when interacting with the hosted LangGraph Studio frontend.

## CLI

By default, calls to most CLI commands log a single analytics event to help improve the CLI experience. Each telemetry event contains the calling process's OS, OS version, Python version, the CLI version, the command name (`dev`, `up`, `run`, etc.), and booleans representing whether a flag was passed to the command. You can see the full analytics logic [here](https://github.com/langchain-ai/langgraph/blob/main/libs/cli/langgraph_cli/analytics.py). 

You can disable all CLI telemetry by setting `LANGGRAPH_CLI_NO_ANALYTICS=1`.

## LangGraph Server (in-memory & docker)

The LangGraph server provides a durable execution runtime that relies on persisting checkpoints of your application state, long-term memories, thread metadata, assistants, and similar resources to a database or file system. Unless you have deliberately customized the storage location, this information is either stored in a postgres database (for `langgraph up` and in deployments) or written to local disk (for `langgraph dev`).

### In-memory development server (`langgraph dev`)

`langgraph dev` runs an in-memory variant of the server as a single Python process. It saves all checkpointing & memory data to disk within a `.langgraph` directory in the current working directory. Apart from the telemetry data described in [CLI only](#cli-only), no data leaves the machine unless you have enabled tracing or your graph code explicitly contacts an external service.

LangSmith tracing is disabled by default, but you can opt-in to tracing by setting `LANGSMITH_TRACING=true` and providing a valid `LANGSMITH_API_KEY` either in your terminal or in the `.env` file indicated by your `langgraph.json` configuration file.

### Single-container (`langgraph up`)

`langgraph up` builds your local package within a docker image and runs the server as 3 containers: the API server, a postgres container, and a redis container. All persistence data (checkpoints, assistants, etc.) are stored in the postgres database. Redis is used as a pubsub connection for real-time streaming of events. You can encrypt all checkpoints before saving to the database by setting a valid `LANGGRAPH_AES_KEY` environment variable.

Additional API calls are made to confirm that the server has a valid license and to track the number of executed runs and tasks. Periodically, the API server validates the provided license key (or API key).

In this configuration, LangSmith tracing is enabled by default. To opt-out, set `LANGSMITH_TRACING=false` in your environment.

If you've disabled tracing, no user data is persisted externally unless your graph code explicitly contacts an external service.

## Studio

LangGraph Studio is a browser client for interacting with your LangGraph server. Though it caches responses in your browser's local storage for performance, it does not trace or persist your private data.

By default, the studio UI collects usage analytics to improve the user experience. This includes:

- Page visits and navigation patterns
- User actions (button clicks)
- Browser type and version
- Screen resolution and viewport size

Importantly, no application data or code (or other sensitive configuration details) are collected.

## Quick reference

In summary, you can opt-out of server-side telemetry by turning off CLI analytics and disabling tracing.

| Variable                       | Purpose                   | Default                          |
| ------------------------------ | ------------------------- | -------------------------------- |
| `LANGGRAPH_CLI_NO_ANALYTICS=1` | Disable CLI analytics     | Analytics enabled                |
| `LANGSMITH_API_KEY`            | Enable LangSmith tracing  | Tracing disabled                 |
| `LANGSMITH_TRACING=false`      | Disable LangSmith tracing | Depends on environment           |

You can opt-out of browser analytics by setting the `DoNotTrack` header in your browser.