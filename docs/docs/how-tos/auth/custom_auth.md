# Add custom authentication

This guide shows how to add custom authentication to your LangGraph Platform application. This guide applies to both LangGraph Platform and self-hosted deployments. It does not apply to isolated usage of the LangGraph open source library in your own custom server.

!!! note

    Custom auth is supported for all **managed LangGraph Platform** deployments, as well as **Enterprise** self-hosted plans. It is not supported for **Lite** self-hosted plans.

## Add custom authentication to your deployment

To leverage custom authentication and access user-level metadata in your deployments, set up custom authentication to automatically populate the `config["configurable"]["langgraph_auth_user"]` object through a custom authentication handler. You can then access this object in your graph with the `langgraph_auth_user` key to [allow an agent to perform authenticated actions on behalf of the user](#enable-agent-authentication).

1. Implement authentication:

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

2. In your `langgraph.json`, add the path to your auth file:

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

3. Once you've set up authentication in your server, requests must include the required authorization information based on your chosen scheme. Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

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

    === "JavaScript Client"

        ```javascript
        import { Client } from "@langchain/langgraph-sdk";

        const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
        const client = new Client({
        apiUrl: "http://localhost:2024",
        defaultHeaders: { Authorization: `Bearer ${my_token}` },
        });
        const threads = await client.threads.search();
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
        const threads = await remoteGraph.invoke(...);
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

## Learn more

* [Authentication & Access Control](../../concepts/auth.md)
* [LangGraph Platform](../../concepts/langgraph_platform.md)
* [Setting up custom authentication tutorial](../../tutorials/auth/getting_started.md)
