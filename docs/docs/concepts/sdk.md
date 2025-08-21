---
search:
  boost: 2
---

# LangGraph SDK

:::python
LangGraph Platform provides a python SDK for interacting with [LangGraph Server](./langgraph_server.md).

!!! tip "Python SDK reference"

    For detailed information about the Python SDK, see [Python SDK reference docs](../cloud/reference/sdk/python_sdk_ref.md).

## Installation

You can install the LangGraph SDK using the following command:

```bash
pip install langgraph-sdk
```

## Python sync vs. async

The Python SDK provides both synchronous (`get_sync_client`) and asynchronous (`get_client`) clients for interacting with LangGraph Server:

=== "Sync"

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url=..., api_key=...)
    client.assistants.search()
    ```

=== "Async"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=..., api_key=...)
    await client.assistants.search()
    ```

## Learn more

- [Python SDK Reference](../cloud/reference/sdk/python_sdk_ref.md)
- [LangGraph CLI API Reference](../cloud/reference/cli.md)
  :::

:::js
LangGraph Platform provides a JS/TS SDK for interacting with [LangGraph Server](./langgraph_server.md).

## Installation

You can add the LangGraph SDK to your project using the following command:

```bash
npm install @langchain/langgraph-sdk
```

## Learn more

- [LangGraph CLI API Reference](../cloud/reference/cli.md)
  :::
