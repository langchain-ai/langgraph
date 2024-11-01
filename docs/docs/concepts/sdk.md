# LangGraph SDK

!!! info "Prerequisites"
    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)

The LangGraph Platform provides both a Python and JS SDK for interacting with the [LangGraph Server API](./langgraph_server.md). 

## Installation

You can install the packages using the appropriate package manager for your language.

=== "Python"
    ```bash
    pip install langgraph-sdk
    ```

=== "JS"
    ```bash
    yarn add @langchain/langgraph-sdk
    ```


## API Reference

You can find the API reference for the SDKs here:

- [Python SDK Reference](../cloud/reference/sdk/python_sdk_ref.md)
- [JS/TS SDK Reference](../cloud/reference/sdk/js_ts_sdk_ref.md)

## Python Sync vs. Async

The Python SDK provides both synchronous (`get_sync_client`) and asynchronous (`get_client`) clients for interacting with the LangGraph Server API.

=== "Async"
    ```python
    from langgraph_sdk import get_client

    client = get_client(url=..., api_key=...)
    await client.assistants.search()
    ```

=== "Sync"

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url=..., api_key=...)
    client.assistants.search()
    ```

## Related

- [LangGraph CLI API Reference](../cloud/reference/cli.md)
- [Python SDK Reference](../cloud/reference/sdk/python_sdk_ref.md)
- [JS/TS SDK Reference](../cloud/reference/sdk/js_ts_sdk_ref.md)