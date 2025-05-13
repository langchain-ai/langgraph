# Configurable Headers

LangGraph allows runtime configuration to modify agent behavior and permissions dynamically. When using the [LangGraph Platform](../quick_start.md), you can pass this configuration in the request body (`config`) or specific request headers. This enables adjustments based on user identity or other request data.

For privacy, control which headers are passed to the runtime configuration via the `http.configurable_headers` section in your `langgraph.json` file.

Here's how to customize the included and excluded headers:

```json
{
  "http": {
    "configurable_headers": {
      "include": ["x-user-id", "x-organization-id", "my-prefix-*"],
      "exclude": ["authorization", "x-api-key"]
    }
  }
}
```


The `include` and `exclude` lists accept exact header names or patterns using `*` to match any number of characters. For your security, no other regex patterns are supported.

## Using within your graph

You can access the included headers in your graph using the `config` argument of any node.

```python
def my_node(state, config):
  organization_id = config["configurable"].get("x-organization-id")
  ...
```

Or by fetching from context (useful in tools and or within other nested functions).

```python
from langgraph.config import get_config

def search_everything(query: str):
  organization_id = get_config()["configurable"].get("x-organization-id")
  ...
```


You can even use this to dynamically compile the graph.

```python
# my_graph.py.
import contextlib

@contextlib.asynccontextmanager
async def generate_agent(config):
  organization_id = config["configurable"].get("x-organization-id")
  if organization_id == "org1":
    graph = ...
    yield graph
  else:
    graph = ...
    yield graph

```

```json
{
  "graphs": {"agent": "my_grph.py:generate_agent"}
}
```

### Opt-out of configurable headers

If you'd like to opt-out of configurable headers, you can simply set a wildcard pattern in the `exclude` list:

```json
{
  "http": {
    "configurable_headers": {
      "exclude": ["*"]
    }
  }
}
```

This will exclude all headers from being added to your run's configuration.

Note that exclusions take precedence over inclusions.