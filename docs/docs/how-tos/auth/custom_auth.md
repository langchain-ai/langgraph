# How to add custom authentication

This guide shows how to add custom authentication to your LangGraph Platform application. This guide applies to both LangGraph Cloud, BYOC, and self-hosted deployments. It does not apply to isolated usage of the LangGraph open source library in your own custom server.

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

      *  [**Authentication & Access Control**](../../concepts/auth.md)
      *  [**LangGraph Platform**](../../concepts/index.md#langgraph-platform)
    
    For a more guided walkthrough, see [**setting up custom authentication**](../../tutorials/auth/getting_started.md) tutorial.

!!! note "Python only"
  
    We currently only support custom authentication and authorization in Python deployments with `langgraph-api>=0.0.11`. Support for LangGraph.JS will be added soon.

## 1. Implement authentication

Create `auth.py` file, with a basic JWT authentication handler:

```python
from langgraph_sdk import Auth

my_auth = Auth()

@my_auth.authenticate
async def authenticate(authorization: str) -> str:
    token = authorization.split(" ", 1)[-1] # "Bearer <token>"
    try:
        # Verify token with your auth provider
        user_id = await verify_token(token)
        return user_id
    except Exception:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid token"
        )

# Optional: Add authorization rules
@my_auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    """Add owner to resource metadata and filter by owner."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

## 2. Update configuration

In your `langgraph.json`, add the path to your auth file:

```json hl_lines="7-9"
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "./auth.py:my_auth"
  }
}
```

## Connect from the client

Once you've set up authentication in your server, requests must include the the required authorization information based on your chosen scheme.
Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

=== "Python Client"

    ```python
    from langgraph_sdk import get_client

    my_token = "your-token" # In practice, you would generate a signed token with your auth provider
    client = get_client(
        url="http://localhost:2024",
        headers={"Authorization": f"Bearer {my_token}"}
    )
    threads = await client.threads.list()
    ```

=== "Python RemoteGraph"

    ```python
    from langgraph.pregel.remote import RemoteGraph
    
    my_token = "your-token" # In practice, you would generate a signed token with your auth provider
    remote_graph = RemoteGraph(
        "agent",
        url="http://localhost:2024",
        headers={"Authorization": f"Bearer {my_token}"}
    )
    threads = await remote_graph.threads.list()
    ```

=== "JavaScript Client"

    ```javascript
    import { Client } from "@langchain/langgraph-sdk";

    const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
    const client = new Client({
      apiUrl: "http://localhost:2024",
      headers: { Authorization: `Bearer ${my_token}` },
    });
    const threads = await client.threads.list();
    ```

=== "JavaScript RemoteGraph"

    ```javascript
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
    const remoteGraph = new RemoteGraph({
      graphId: "agent",
      url: "http://localhost:2024",
      headers: { Authorization: `Bearer ${my_token}` },
    });
    const threads = await remoteGraph.threads.list();
    ```

=== "CURL"

    ```bash
    curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
    ```