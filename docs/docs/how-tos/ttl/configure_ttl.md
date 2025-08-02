# How to add TTLs to your LangGraph application

!!! tip "Prerequisites"

    This guide assumes familiarity with the [LangGraph Platform](../../concepts/langgraph_platform.md), [Persistence](../../concepts/persistence.md), and [Cross-thread persistence](../../concepts/persistence.md#memory-store) concepts.

???+ note "LangGraph platform only"
    
    TTLs are only supported for LangGraph platform deployments. This guide does not apply to LangGraph OSS.

The LangGraph Platform persists both [checkpoints](../../concepts/persistence.md#checkpoints) (thread state) and [cross-thread memories](../../concepts/persistence.md#memory-store) (store items). Configure Time-to-Live (TTL) policies in `langgraph.json` to automatically manage the lifecycle of this data, preventing indefinite accumulation.

## Configuring Checkpoint TTL

Checkpoints capture the state of conversation threads. Setting a TTL ensures old checkpoints and threads are automatically deleted.

Add a `checkpointer.ttl` configuration to your `langgraph.json` file:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200 
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200 
    }
  }
}
```
:::

*   `strategy`: Specifies the action taken on expiration. Currently, only `"delete"` is supported, which deletes all checkpoints in the thread upon expiration.
*   `sweep_interval_minutes`: Defines how often, in minutes, the system checks for expired checkpoints.
*   `default_ttl`: Sets the default lifespan of checkpoints in minutes (e.g., 43200 minutes = 30 days).

## Configuring Store Item TTL

Store items allow cross-thread data persistence. Configuring TTL for store items helps manage memory by removing stale data.

Add a `store.ttl` configuration to your `langgraph.json` file:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

*   `refresh_on_read`: (Optional, default `true`) If `true`, accessing an item via `get` or `search` resets its expiration timer. If `false`, TTL only refreshes on `put`.
*   `sweep_interval_minutes`: (Optional) Defines how often, in minutes, the system checks for expired items. If omitted, no sweeping occurs.
*   `default_ttl`: (Optional) Sets the default lifespan of store items in minutes (e.g., 10080 minutes = 7 days). If omitted, items do not expire by default.

## Combining TTL Configurations

You can configure TTLs for both checkpoints and store items in the same `langgraph.json` file to set different policies for each data type. Here is an example:

:::python
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

:::js
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.ts:graph"
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 60,
      "default_ttl": 43200
    }
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 120,
      "default_ttl": 10080
    }
  }
}
```
:::

## Runtime Overrides

The default `store.ttl` settings from `langgraph.json` can be overridden at runtime by providing specific TTL values in SDK method calls like `get`, `put`, and `search`.

## Deployment Process

After configuring TTLs in `langgraph.json`, deploy or restart your LangGraph application for the changes to take effect. Use `langgraph dev` for local development or `langgraph up` for Docker deployment.

See the @[langgraph.json CLI reference][langgraph.json] for more details on the other configurable options.