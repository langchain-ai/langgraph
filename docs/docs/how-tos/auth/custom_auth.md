# Add custom authentication

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

      *  [**Authentication & Access Control**](../../concepts/auth.md)
      *  [**LangGraph Platform**](../../concepts/langgraph_platform.md)

    For a more guided walkthrough, see [**setting up custom authentication**](../../tutorials/auth/getting_started.md) tutorial.

???+ note "Support by deployment type"

    Custom auth is supported for all deployments in the **managed LangGraph Platform**, as well as **Enterprise** self-hosted plans.

This guide shows how to add custom authentication to your LangGraph Platform application. This guide applies to both LangGraph Platform and self-hosted deployments. It does not apply to isolated usage of the LangGraph open source library in your own custom server.

!!! note

    Custom auth is supported for all **managed LangGraph Platform** deployments, as well as **Enterprise** self-hosted plans.

## Add custom authentication to your deployment

To leverage custom authentication and access user-level metadata in your deployments, set up custom authentication to automatically populate the `config["configurable"]["langgraph_auth_user"]` object through a custom authentication handler. You can then access this object in your graph with the `langgraph_auth_user` key to [allow an agent to perform authenticated actions on behalf of the user](#enable-agent-authentication).

:::python

1.  Implement authentication:

    !!! note

        Without a custom `@auth.authenticate` handler, LangGraph sees only the API-key owner (usually the developer), so requests aren’t scoped to individual end-users. To propagate custom tokens, you must implement your own handler.

    ```python
    from langgraph_sdk import Auth
    import requests

    auth = Auth()

    def is_valid_key(api_key: str) -> bool:
        is_valid = # your API key validation logic
        return is_valid

    @auth.authenticate # (1)!
    async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
        api_key = headers.get("x-api-key")
        if not api_key or not is_valid_key(api_key):
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API key")

        # Fetch user-specific tokens from your secret store
        user_tokens = await fetch_user_tokens(api_key)

        return { # (2)!
            "identity": api_key,  #  fetch user ID from LangSmith
            "github_token" : user_tokens.github_token
            "jira_token" : user_tokens.jira_token
            # ... custom fields/secrets here
        }
    ```

    1. This handler receives the request (headers, etc.), validates the user, and returns a dictionary with at least an identity field.
    2. You can add any custom fields you want (e.g., OAuth tokens, roles, org IDs, etc.).

2.  In your `langgraph.json`, add the path to your auth file:

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

3.  Once you've set up authentication in your server, requests must include the required authorization information based on your chosen scheme. Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

    === "Python Client"

        ```python
        from langgraph_sdk import get_client

        my_token = "your-token" # In practice, you would generate a signed token with your auth provider
        client = get_client(
            url="http://localhost:2024",
            headers={"Authorization": f"Bearer {my_token}"}
        )
        threads = await client.threads.search()
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
        threads = await remote_graph.ainvoke(...)
        ```
        ```python
        from langgraph.pregel.remote import RemoteGraph

        my_token = "your-token" # In practice, you would generate a signed token with your auth provider
        remote_graph = RemoteGraph(
            "agent",
            url="http://localhost:2024",
            headers={"Authorization": f"Bearer {my_token}"}
        )
        threads = await remote_graph.ainvoke(...)
        ```

    === "CURL"

        ```bash
        curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
        ```

## Enable agent authentication

After [authentication](#add-custom-authentication-to-your-deployment), the platform creates a special configuration object (`config`) that is passed to LangGraph Platform deployment. This object contains information about the current user, including any custom fields you return from your `@auth.authenticate` handler.

To allow an agent to perform authenticated actions on behalf of the user, access this object in your graph with the `langgraph_auth_user` key:

```python
def my_node(state, config):
    user_config = config["configurable"].get("langgraph_auth_user")
    # token was resolved during the @auth.authenticate function
    token = user_config.get("github_token","")
    ...
```

!!! note

    Fetch user credentials from a secure secret store. Storing secrets in graph state is not recommended.

### Authorizing a Studio user

By default, if you add custom authorization on your resources, this will also apply to interactions made from the Studio. If you want, you can handle logged-in Studio users differently by checking [is_studio_user()](../../reference/functions/sdk_auth.isStudioUser.html).

!!! note
    `is_studio_user` was added in version 0.1.73 of the langgraph-sdk. If you're on an older version, you can still check whether `isinstance(ctx.user, StudioUser)`.

```python
from langgraph_sdk.auth import is_studio_user, Auth
auth = Auth()

# ... Setup authenticate, etc.

@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict  # The payload being sent to this access method
) -> dict:  # Returns a filter dict that restricts access to resources
    if is_studio_user(ctx.user):
        return {}

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

Only use this if you want to permit developer access to a graph deployed on the managed LangGraph Platform SaaS.

:::

:::js

1.  Implement authentication:

    !!! note

        Without a custom `authenticate` handler, LangGraph sees only the API-key owner (usually the developer), so requests aren’t scoped to individual end-users. To propagate custom tokens, you must implement your own handler.

    ```typescript
    import { Auth, HTTPException } from "@langchain/langgraph-sdk/auth";

    const auth = new Auth()
      .authenticate(async (request) => {
        const authorization = request.headers.get("Authorization");
        const token = authorization?.split(" ")[1]; // "Bearer <token>"
        if (!token) {
          throw new HTTPException(401, "No token provided");
        }
        try {
          const user = await verifyToken(token);
          return user;
        } catch (error) {
          throw new HTTPException(401, "Invalid token");
        }
      })
      // Add authorization rules to actually control access to resources
      .on("*", async ({ user, value }) => {
        const filters = { owner: user.identity };
        const metadata = value.metadata ?? {};
        metadata.update(filters);
        return filters;
      })
      // Assumes you organize information in store like (user_id, resource_type, resource_id)
      .on("store", async ({ user, value }) => {
        const namespace = value.namespace;
        if (namespace[0] !== user.identity) {
          throw new HTTPException(403, "Not authorized");
        }
      });
    ```

    1. This handler receives the request (headers, etc.), validates the user, and returns an object with at least an identity field.
    2. You can add any custom fields you want (e.g., OAuth tokens, roles, org IDs, etc.).

2.  In your `langgraph.json`, add the path to your auth file:

    ```json hl_lines="7-9"
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./agent.ts:graph"
      },
      "env": ".env",
      "auth": {
        "path": "./auth.ts:my_auth"
      }
    }
    ```

3.  Once you've set up authentication in your server, requests must include the required authorization information based on your chosen scheme. Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

    === "SDK Client"

        ```javascript
        import { Client } from "@langchain/langgraph-sdk";

        const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
        const client = new Client({
          apiUrl: "http://localhost:2024",
          defaultHeaders: { Authorization: `Bearer ${my_token}` },
        });
        const threads = await client.threads.search();
        ```

    === "RemoteGraph"

        ```javascript
        import { RemoteGraph } from "@langchain/langgraph/remote";

        const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
        const remoteGraph = new RemoteGraph({
        graphId: "agent",
          url: "http://localhost:2024",
          headers: { Authorization: `Bearer ${my_token}` },
        });
        const threads = await remoteGraph.invoke(...);
        ```

    === "CURL"

        ```bash
        curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
        ```

## Enable agent authentication

After [authentication](#add-custom-authentication-to-your-deployment), the platform creates a special configuration object (`config`) that is passed to LangGraph Platform deployment. This object contains information about the current user, including any custom fields you return from your `authenticate` handler.

To allow an agent to perform authenticated actions on behalf of the user, access this object in your graph with the `langgraph_auth_user` key:

```ts
async function myNode(state, config) {
  const userConfig = config["configurable"]["langgraph_auth_user"];
  // token was resolved during the authenticate function
  const token = userConfig["github_token"];
  ...
}
```

!!! note

    Fetch user credentials from a secure secret store. Storing secrets in graph state is not recommended.

:::

## Learn more

- [Authentication & Access Control](../../concepts/auth.md)
- [LangGraph Platform](../../concepts/langgraph_platform.md)
- [Setting up custom authentication tutorial](../../tutorials/auth/getting_started.md)
