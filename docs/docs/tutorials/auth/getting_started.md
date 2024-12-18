# Setting up Custom Authentication

In this tutorial, we will build a chatbot that only lets specific users access it. We'll start with the LangGraph template and add token-based security step by step. By the end, you'll have a working chatbot that checks for valid tokens before allowing access.

!!! note "This is part 1 of our authentication series:"

    1. Basic Authentication (you are here) - Control who can access your bot
    2. [Resource Authorization](resource_auth.md) - Let users have private conversations
    3. [Production Auth](add_auth_server.md) - Add real user accounts and validate using OAuth2

## Setting up our project

First, let's create a new chatbot using the LangGraph starter template:

```bash
pip install -U "langgraph-cli[inmem]"
langgraph new --template=new-langgraph-project-python custom-auth
cd custom-auth
```

The template gives us a placeholder LangGraph app. Let's try it out by installing the local dependencies and running the development server.
```shell
pip install -e .
langgraph dev
```
> - 🚀 API: http://127.0.0.1:2024
> - 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
> - 📚 API Docs: http://127.0.0.1:2024/docs
> 
> This in-memory server is designed for development and testing.
> For production use, please use LangGraph Cloud.

If everything works, the server should start and open the studio in your browser.

Now that we've seen the base LangGraph app, let's add authentication to it!

## Adding Authentication

The [`Auth`](../../cloud/reference/sdk/python_sdk_ref.md#langgraph_sdk.auth.Auth) object lets you register an authentication function that the LangGraph platform will run on every request. This function receives each request and decides whether to accept or reject.

Create a new file `src/security/auth.py`. This is where we'll our code will live to check if users are allowed to access our bot:

```python
from langgraph_sdk import Auth

# This is our toy user database
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

auth = Auth()


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Check if the user's token is valid."""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"
    # Check if token is valid
    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    # Return user info if valid
    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }
```

Notice that our authentication handler does two important things:

1. Checks if a valid token is provided
2. Returns the user's 

Now tell LangGraph to use our authentication by adding the following to the `langgraph.json` configuration:

```json
{
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

## Testing Our Secure Bot

Let's start the server again to test everything out!

```bash
langgraph dev --no-browser
```

??? note "Custom auth in the studio"

    If you didn't add the `--no-browser`, the studio UI will open in the browser. You may wonder, how is the studio able to still connect to our server? By default, we also permit access from the LangGraph studio, even when using custom auth. This makes it easier to develop and test your bot in the studio. You can remove this alternative authentication option by
    setting `disable_studio_auth: "true"` in your auth configuration:
    ```json
    {
        "auth": {
        "path": "src/security/auth.py:auth",
        "disable_studio_auth": "true"
        }
    }
    ```

Now let's try to chat with our bot. Create a new file `test_auth.py`:

```python
import asyncio
from langgraph_sdk import get_client


async def test_auth():
    # Try without a token (should fail)
    client = get_client(url="http://localhost:2024")
    try:
        thread = await client.threads.create()
        print("❌ Should have failed without token!")
    except Exception as e:
        print("✅ Correctly blocked access:", e)

    # Try with a valid token
    client = get_client(
        url="http://localhost:2024", headers={"Authorization": "Bearer user1-token"}
    )

    # Create a thread and chat
    thread = await client.threads.create()
    print(f"✅ Created thread as Alice: {thread['thread_id']}")

    response = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Hello!"}]},
    )
    print("✅ Bot responded:")
    print(response)


if __name__ == "__main__":
    asyncio.run(test_auth())
```

Run the test code and you should see that:
1. Without a valid token, we can't access the bot
2. With a valid token, we can create threads and chat

Congratulations! You've built a chatbot that only lets "authorized" users access it. While this system doesn't (yet) implement a production-ready security scheme, we've learned the basic mechanics of how to control access to our bot. In the next tutorial, we'll learn how to give each user their own private conversations.

## What's Next?

Now that you can control who accesses your bot, you might want to:
1. Move on to [Resource Authorization](resource_auth.md) to learn how to make conversations private
2. Read more about [authentication concepts](../../concepts/auth.md)
3. Check out the [API reference](../../cloud/reference/sdk/python_sdk_ref.md) for more authentication options