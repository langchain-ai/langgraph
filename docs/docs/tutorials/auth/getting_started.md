# Setting up custom authentication

Let's add OAuth2 token authentication to a LangGraph template. This lets users interact with our bot making their conversations accessible to other users. This tutorial covers the core concepts of token-based authentication and show how to integrate with an authentication server.

???+ tip "Prerequisites"
This guide assumes familiarity with the following concepts:

      *  The [LangGraph Platform](../../concepts/index.md#langgraph-platform)
      *  [Authentication & Access Control](../../concepts/auth.md) in the LangGraph Platform


    Before you begin, ensure you have the following:

      * [GitHub account](https://github.com/)
      * [LangSmith account](https://smith.langchain.com/)
      * [Supabase account](https://supabase.com/)
      * [Anthropic API key](https://console.anthropic.com/)

??? note "Python only"

    We currently only support custom authentication and authorization in Python deployments with `langgraph-api>=0.0.11`. Support for LangGraph.JS will be added soon.

??? tip "Default authentication"
When deploying to LangGraph Cloud, requests are authenticated using LangSmith API keys by default. This gates access to the server but doesn't provide fine-grained access control over threads. Self-hosted LangGraph platform has no default authentication. This guide shows how to add custom authentication handlers that work in both cases, to provide fine-grained access control over threads, runs, and other resources.

## Overview

The key components in a token-based authentication system are:

1. **Auth server**: manages users and generates signed tokens (could be Supabase, Auth0, or your own server)
2. **Client**: gets tokens from auth server and includes them in requests. This is typically the user's browser or mobile app.
3. **LangGraph backend**: validates tokens and enforces access control to control access to your agents and data.

For a typical interaction:

1. User authenticates with the auth server (username/password, OAuth, "Sign in with Google", etc.)
2. Auth server returns a signed JWT token attesting "I am user X with claims/roles Y"
3. User includes this token in request headers to LangGraph
4. LangGraph validates token signature and checks claims against the auth server. If valid, it allows the request, using custom filters to restrict access only to the user's resources.


## 1. Clone the template

Clone the [LangGraph template](https://github.com/langchain-ai/new-langgraph-project) to get started.

```shell
pip install -U "langgraph-cli[inmem]"
langgraph new --template=new-langgraph-project-python custom-auth
cd custom-auth
```

### 2. Set up environment variables

Copy the example `.env` file and add your Supabase credentials.

```bash
cp .env.example .env
```

Add to your `.env`:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key # aka the service_role secret
ANTHROPIC_API_KEY=your-anthropic-key  # For the LLM in our chatbot
```

To get your Supabase credentials:

1. Create a project at [supabase.com](https://supabase.com)
2. Go to Project Settings > API
3. Add these credentials to your `.env` file:

Also note down your project's "anon public" key. We'll use this for client authentication below.

### 3. Create the auth handler

Now we'll create an authentication handler that does two things:
1. Authenticates users by validating their tokens (`@auth.authenticate`)
2. Controls what resources those users can access (`@auth.on`)

We'll use the `Auth` class from `langgraph_sdk` to register these handler functions. The LangGraph backend will automatically call these functions that you've registered whenever a user makes a request.

Create a new file at `src/security/auth.py`:

```python
import os
import httpx
from langgraph_sdk import Auth

# These will be loaded from your .env file in the next step
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# The auth handler registers functions that the LangGraph backend will call
auth = Auth()

@auth.authenticate
async def get_current_user(
    authorization: str | None,  # "Bearer <token>"
) -> tuple[list[str], Auth.types.MinimalUserDict]:
    """Verify the JWT token and return user info."""
    try:
        # Fetch the user info from Supabase
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": authorization,
                    "apiKey": SUPABASE_SERVICE_KEY,
                },
            )
            assert response.status_code == 200
            user_data = response.json()
            return {
                "identity": user_data["id"],
                "display_name": user_data.get("name"),
                "is_authenticated": True,
            }
    except Exception as e:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid token"
        )
```

This handler validates the user's information, but by itself doesn't restrict what authenticated users can access. Let's add an authorization handler to limit access to resources. We'll do this by:
1. Adding the user's ID to resource metadata when they create something
2. Using that metadata to filter what resources they can see

Register this authorization handler with the `@auth.on` decorator. This function will run on all calls that make it past the authentication stage.

```python
@auth.on
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

Now when users create threads, assistants, runs, or other resources, their ID is automatically added as the owner in its metadata, and they can only see the threads they own.

### 3. Configure `langgraph.json`

Next, we need to tell LangGraph that we've created an auth handler. Open `langgraph.json` and add:

```json
{
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

This points LangGraph to our `auth` object in the `auth.py` file.

### 4. Start the server

Install dependencies and start LangGraph:

```shell
pip install -e .
langgraph dev --no-browser
```

## Interacting with the server

First, let's set up our environment and helper functions. Fill in the values for your Supabase anon key, and provide a working email address for our test users.

!!! tip "Multiple example emails"
You can create multiple users with a shared email bya dding a "+" to the email address. For example, "myemail@gmail.com" can be used to create "myemail+1@gmail.com" and "myemail+2@gmail.com".

Copy the code below. Make sure to fill out the Supabase URL & anon key, as well as the email addresses for your test users. Then run the code.

```python
import os
import httpx
import dotenv

from langgraph_sdk import get_client

supabase_url: str = "CHANGEME"
supabase_anon_key: str = "CHANGEME"  # Your project's anon/public key
user_1_email = "CHANGEME"  # Your test email
user_2_email = "CHANGEME"  # A second test email
password = "password"  # Very secure! :)

# Helper functions for authentication
async def sign_up(email, password):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{supabase_url}/auth/v1/signup",
            headers={
                "apikey": supabase_anon_key,
                "Content-Type": "application/json",
            },
            json={
                "email": email,
                "password": password
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Sign up failed:", response.status_code, response.text)

async def login(email, password):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{supabase_url}/auth/v1/token?grant_type=password",
            headers={
                "apikey": supabase_anon_key,
                "Content-Type": "application/json",
            },
            json={
                "email": email,
                "password": password
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError("Login failed:", response.status_code, response.text)
```

Now let's create two test users:

```python
# Create our test users
await sign_up(user_1_email, password)
await sign_up(user_2_email, password)
```

⚠️ Before continuing: Check your email for both addresses and click the confirmation links. Don't worry about any error pages you might see from the confirmation redirect - those would normally be handled by your frontend.

Now let's log in as our first user and create a thread:

```python
# Log in as user 1
user_1_login_data = await login(user_1_email, password)
user_1_token = user_1_login_data["access_token"]

# Create an authenticated client
client = get_client(
    url="http://localhost:2024",
    headers={"Authorization": f"Bearer {user_1_token}"}
)

# Create a thread and chat with the bot
thread = await client.threads.create()
print(f'Created thread: {thread["thread_id"]}')

# Have a conversation
async for event, (chunk, metadata) in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Tell me a short joke"}]},
    stream_mode="messages-tuple",
):
    if event == "messages" and metadata["langgraph_node"] == "chatbot":
        print(chunk['content'], end="", flush=True)

# View the thread history
thread = await client.threads.get(thread["thread_id"])
print(f"\nThread:\n{thread}")
```

We were able to create a thread and have a conversation with the bot. Great!

Now let's see what happens when we try to access the server without authentication:

```python
# Try to access without a token
unauthenticated_client = get_client(url="http://localhost:2024")
try:
    await unauthenticated_client.threads.create()
except Exception as e:
    print(f"Failed without token: {e}")  # Will show 403 Forbidden
```

Without an authentication token, we couldn't create a new thread!

If we try to access a thread owned by another user, we'll get an error:

```python
# Log in as user 2
user_2_login_data = await login(user_2_email, password)
user_2_token = user_2_login_data["access_token"]

# Create client for user 2
user_2_client = get_client(
    url="http://localhost:2024",
    headers={"Authorization": f"Bearer {user_2_token}"}
)

# This passes
thread2 = await unauthenticated_client.threads.create()

# Try to access user 1's thread
try:
    await user_2_client.threads.get(thread["thread_id"])
except Exception as e:
    print(f"Failed to access other user's thread: {e}")  # Will show 404 Not Found
```

Notice that:

1. With a valid token, we can create and interact with threads
2. Without a token, we get an authentication error saying we are forbidden
3. Even with a valid token, users can only access their own threads

## Deploying to LangGraph Cloud

Now that you've set everything up, you can deploy your LangGraph application to LangGraph Cloud! Simply:

1. Push your code to a new github repository.
2. Navigate to the LangGraph Platform page and click "+New Deployment".
3. Connect to your GitHub repository and copy the contents of your `.env` file as environment variables.
4. Click "Submit".

Once deployed, you should be able to run the client code above again, replacing the `http://localhost:2024` with the URL of your deployment.

## Next steps

Now that you understand token-based authentication, you can try integrating this in actual frontend code! You can see a longer example of this tutorial at the [custom auth template](https://github.com/langchain-ai/custom-auth). There, you can see a full end-to-end example of adding custom authentication to a LangGraph chatbot using a react web frontend. 