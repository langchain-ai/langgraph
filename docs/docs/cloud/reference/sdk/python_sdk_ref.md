# Python SDK Reference

The Python SDK provides four underlying clients (`AssistantsClient`, `ThreadsClient`, `RunsClient`, `CronClient`) that correspond to each of the core API models and one top-level client (`LangGraphClient`) to access them.

## get_client()

The `get_client()` function returns the top-level `LangGraphClient` client.

```python
from langgraph_sdk import get_client

# get top-level LangGraphClient
client = get_client(url="http://localhost:8123")

# example usage: client.<model>.<method_name>()
assistants = await client.assistants.get(assistant_id="some_uuid")
```

::: langgraph_sdk.client.get_client
    handler: python

## LangGraphClient

`LangGraphClient` is the top-level client for accessing `AssistantsClient`, `ThreadsClient`, `RunsClient`, and `CronClient`.

::: langgraph_sdk.client.LangGraphClient
    handler: python

## AssistantsClient

Access the `AssistantsClient` via the `LangGraphClient.assistants` attribute.
```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
await client.assistants.<method_name>()
```

::: langgraph_sdk.client.AssistantsClient
    handler: python

## ThreadsClient

Access the `ThreadsClient` via the `LangGraphClient.threads` attribute.
```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
await client.threads.<method_name>()
```

::: langgraph_sdk.client.ThreadsClient
    handler: python

## RunsClient

Access the `RunsClient` via the `LangGraphClient.runs` attribute.
```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
await client.runs.<method_name>()
```

::: langgraph_sdk.client.RunsClient
    handler: python

## CronClient

Access the `CronClient` via the `LangGraphClient.crons` attribute.
```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")
await client.crons.<method_name>()
```

::: langgraph_sdk.client.CronClient
    handler: python
