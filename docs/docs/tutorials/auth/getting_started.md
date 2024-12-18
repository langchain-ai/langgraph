# Setting up custom authentication

Let's add custom authentication to a LangGraph template. This lets users interact with our bot making their conversations accessible to other users. This tutorial covers the core concepts of token-based authentication and show how to integrate with an authentication server.

??? note "Default authentication"
When deploying to LangGraph Cloud, requests are authenticated using LangSmith API keys by default. This gates access to the server but doesn't provide fine-grained access control over threads. Self-hosted LangGraph platform has no default authentication. This guide shows how to add custom authentication handlers that work in both cases, to provide fine-grained access control over threads, runs, and other resources.

!!! note "Prerequisites"

    Before you begin, ensure you have the following:
    - [GitHub account](https://github.com/)
    - [LangSmith account](https://smith.langchain.com/)
    - [Supabase account](https://supabase.com/)
    - [Anthropic API key](https://console.anthropic.com/)

## Understanding authentication flow

The key components in a token-based authentication system are:

1. **Auth server**: manages users and generates signed tokens (could be Supabase, Auth0, or your own server)
2. **Client**: gets tokens from auth server and includes them in requests. This is typically the user's browser or mobile app.
3. **LangGraph backend**: validates tokens and enforces access control to control access to your agents and data.

Here's how it typically works:

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

### 2. Create the auth handler

Next, let's create our authentication handler. Create a new file at `src/security/auth.py`:

```python
import os
import httpx
from langgraph_sdk import Auth

# Load from your .env file
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Create the auth object we'll use to protect our endpoints
auth = Auth()

@auth.authenticate
async def get_current_user(
    authorization: str | None,  # "Bearer <token>"
) -> tuple[list[str], Auth.types.MinimalUserDict]:
    """Verify the JWT token and return user info."""
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
            if response.status_code != 200:
                raise Auth.exceptions.HTTPException(
                    status_code=401,
                    detail="Invalid token"
                )

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

This handler ensures only users with valid tokens can access our server. However, all users can still see each other's threads. Let's fix that by adding an authorization filter to the bottom of `auth.py`:

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

Now when users create threads, their ID is automatically added as the owner, and they can only see threads they own.

### 2. Configure LangGraph

Next, tell LangGraph about our auth handler. Open `langgraph.json` and add:

```json
{
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

This points LangGraph to our `auth` object in the `auth.py` file.

### 3. Set up environment variables

Copy the example env file and add your Supabase credentials. To get your Supabase credentials:

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to Project Settings > API to find your project's credentials
3. Add these credentials to your `.env` file:

```bash
cp .env.example .env
```

Add to your `.env`:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key # aka the service_role secret
ANTHROPIC_API_KEY=your-anthropic-key  # For the LLM in our chatbot
```

Also note down your project's "anon public" key - we'll use this for client authentication below.

### 4. Start the server

Install dependencies and start LangGraph:

```bash
pip install -e .
langgraph dev --no-browser
```

## Interacting with the server

First, let's set up our environment and helper functions. Fill in the values for your Supabase anon key, and provide a working email address for our test users. You can use a single email with "+" to create multiple users, e.g. "myemail+1@gmail.com" and "myemail+2@gmail.com".

```python
import os
import httpx
import dotenv

from langgraph_sdk import get_client

dotenv.load_dotenv()

supabase_url: str = os.environ.get("SUPABASE_URL")
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

Now let's see what happens when we try to access without authentication:

```python
# Try to access without a token
unauthenticated_client = get_client(url="http://localhost:2024")
try:
    await unauthenticated_client.threads.create()
except Exception as e:
    print(f"Failed without token: {e}")  # Will show 403 Forbidden
```

Finally, let's try accessing user 1's thread as user 2:

```python
# Log in as user 2
user_2_login_data = await login(user_2_email, password)
user_2_token = user_2_login_data["access_token"]

# Create client for user 2
user_2_client = get_client(
    url="http://localhost:2024",
    headers={"Authorization": f"Bearer {user_2_token}"}
)

# Try to access user 1's thread
try:
    await user_2_client.threads.get(thread["thread_id"])
except Exception as e:
    print(f"Failed to access other user's thread: {e}")  # Will show 404 Not Found
```

This demonstrates that:

1. With a valid token, we can create and interact with threads
2. Without a token, we get a 401 Unauthorized or 403 Forbidden error
3. Even with a valid token, users can only access their own threads

## Deploying to LangGraph Cloud

Now that you've set everything up, you can deploy your LangGraph application to LangGraph Cloud! Simply:

1. Push your code to a new github repository.
2. Navigate to the LangGraph Platform page and click "+New Deployment".
3. Connect to your GitHub repository and copy the contents of your `.env` file as environment variables.
4. Click "Submit".

Once deployed, you should be able to run the code above, replacing the `http://localhost:2024` with the URL of your deployment.

## Next steps

Now that you understand token-based authentication:

1. Add password hashing and secure user management
2. Add user-specific resource ownership (see [resource access control](./resource_access.md))
3. Implement more advanced auth patterns
