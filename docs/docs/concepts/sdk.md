---
search:
  boost: 2
---

# LangGraph SDK

LangGraph Platform provides both a Python SDK for interacting with [LangGraph Server](./langgraph_server.md).

!!! tip "Python SDK reference"
  
    For detailed information about the Python SDK, see [Python SDK reference docs](../cloud/reference/sdk/python_sdk_ref.md).

## Installation

You can install the packages using the appropriate package manager for your language:

=== "Python"
    ```bash
    pip install langgraph-sdk
    ```

=== "JS"
    ```bash
    yarn add @langchain/langgraph-sdk
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
- [JS/TS SDK Reference](../cloud/reference/sdk/js_ts_sdk_ref.md)