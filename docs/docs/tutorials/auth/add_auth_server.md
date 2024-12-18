# Connecting an Authentication Provider

In the previous tutorial, we added [resource authorization](../../concepts/auth.md#resource-authorization) to give users private conversations. However, we were still using hard-coded tokens for authentication, which is not secure. Now we'll replace those tokens with real user accounts using [OAuth2](../../concepts/auth.md#oauth2-authentication).

We'll keep the same [`Auth`](../../cloud/reference/sdk/python_sdk_ref.md#langgraph_sdk.auth.Auth) object and [resource-level access control](../../concepts/auth.md#resource-level-access-control), but upgrade our authentication to use Supabase as our identity provider. While we use Supabase in this tutorial, the concepts apply to any OAuth2 provider. You'll learn how to:

1. Replace test tokens with real [JWT tokens](../../concepts/auth.md#jwt-tokens)
2. Integrate with OAuth2 providers for secure user authentication
3. Handle user sessions and metadata while maintaining our existing authorization logic

!!! note "This is part 3 of our authentication series:"

    1. [Basic Authentication](getting_started.md) - Control who can access your bot
    2. [Resource Authorization](resource_auth.md) - Let users have private conversations
    3. Production Auth (you are here) - Add real user accounts and validate using OAuth2

!!! warning "Prerequisites"

    - [Create a Supabase project](https://supabase.com/dashboard)
    - Have your project URL and service role key ready

## Background

OAuth2 involves three main roles:

1. **Authorization server**: The identity provider (e.g., Supabase, Auth0, Google) that handles user authentication and issues tokens
2. **Application backend**: Your LangGraph application. This validates tokens and serves protected resources (conversation data)
3. **Client application**: The web or mobile app where users interact with your service

A standard OAuth2 flow works something like this:


```mermaid
sequenceDiagram
    participant User
    participant Client
    participant AuthServer
    participant LangGraph Backend

    User->>Client: Initiate login
    User->>AuthServer: Enter credentials
    AuthServer->>Client: Send tokens
    Client->>LangGraph Backend: Request with token
    LangGraph Backend->>AuthServer: Validate token
    AuthServer->>LangGraph Backend: Token valid
    LangGraph Backend->>Client: Serve request (e.g., run agent or graph)
```

In the following example, we'll use Supabase as our auth server. The LangGraph application will provide the backend for your app, and we will write test code for the client app.
Let's get started!

## Setting Up Authentication Provider

First, let's install the required dependencies. Start in your `custom-auth` directory and ensure you have the `langgraph-cli` installed:

```bash
cd custom-auth
pip install -U "langgraph-cli[inmem]"
```

Next, we'll need to fech the URL of our auth server and the private key for authentication.
Since we're using Supabase for this, we can do this in the Supabase dashboard:

1. In the left sidebar, click on t️⚙ Project Settings" and then click "API"
2. Copy your project URL and add it to your `.env` file

```shell
echo "SUPABASE_URL=your-project-url" >> .env
```

3. Next, copy your service role secret key and add it to your `.env` file

```shell
echo "SUPABASE_SERVICE_KEY=your-service-role-key" >> .env
```

4. Finally, copy your "anon public" key and note it down. This will be used later when we set up our client code.

```bash
SUPABASE_URL=your-project-url
SUPABASE_SERVICE_KEY=your-service-role-key
```

## Implementing Token Validation

In the previous tutorials, we used the [`Auth`](../../cloud/reference/sdk/python_sdk_ref.md#langgraph_sdk.auth.Auth) object to:

1. Validate hard-coded tokens in the [authentication tutorial](getting_started.md)
2. Add resource ownership in the [authorization tutorial](resource_auth.md)

Now we'll upgrade our authentication to validate real JWT tokens from Supabase. The key changes will all be in the [`@auth.authenticate`](../../cloud/reference/sdk/python_sdk_ref.md#langgraph_sdk.auth.Auth.authenticate) decorated function:

1. Instead of checking against a hard-coded list of tokens, we'll make an HTTP request to Supabase to validate the token
2. We'll extract real user information (ID, email) from the validated token

And we'll keep our existing resource authorization logic unchanged

Let's update `src/security/auth.py` to implement this:

```python
import os
import httpx
from langgraph_sdk import Auth

auth = Auth()

# This is loaded from the `.env` file you created above
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@auth.authenticate
async def get_current_user(authorization: str | None):
    """Validate JWT tokens and extract user information."""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    try:
        # Verify token with auth provider
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": authorization,
                    "apiKey": SUPABASE_SERVICE_KEY,
                },
            )
            assert response.status_code == 200
            user = response.json()
            return {
                "identity": user["id"],  # Unique user identifier
                "email": user["email"],
                "is_authenticated": True,
            }
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))


# Keep our resource authorization from the previous tutorial
@auth.on
async def add_owner(ctx, value):
    """Make resources private to their creator using resource metadata."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

The most important change is that we're now validating tokens with a real authentication server. Our authentication handler has the private key for our Supabase project, which we can use to validate the user's token and extract their information.

Let's test this with a real user account!

## Testing Authentication Flow

Create a new file `create_users.py`. This will stand-in for a frontend that lets users sign up and log in.

```python
import argparse
import asyncio
import os

import dotenv
import httpx
from langgraph_sdk import get_client

dotenv.load_dotenv()

# Get email from command line
parser = argparse.ArgumentParser()
parser.add_argument("email", help="Your email address for testing")
args = parser.parse_args()

base_email = args.email.split("@")
email1 = f"{base_email[0]}+1@{base_email[1]}"
email2 = f"{base_email[0]}+2@{base_email[1]}"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


async def sign_up(email: str, password: str):
    """Create a new user account."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            json={"email": email, "password": password},
            headers={"apiKey": SUPABASE_SERVICE_KEY},
        )
        assert response.status_code == 200
        return response.json()

async def main():
    # Create two test users
    password = "secure-password"  # CHANGEME
    print(f"Creating test users: {email1} and {email2}")
    await sign_up(email1, password)
    await sign_up(email2, password)

if __name__ == "__main__":
    asyncio.run(main())
```

Then run the setup script:

```shell
python create_users.py CHANGEME@example.com
```

!!! tip "About test emails"
    We'll create two test accounts by adding "+1" and "+2" to your email. For example, if you use "myemail@gmail.com", we'll create "myemail+1@gmail.com" and "myemail+2@gmail.com". All emails will be delivered to your original address.

⚠️ Before continuing: Check your email and click both confirmation links. This would normally be handled by your frontend.

Now let's test that users can only see their own data. Create a new file `test_oauth.py`. This will stand-in for your application's frontend.

```python
import argparse
import asyncio
import os

import dotenv
import httpx
from langgraph_sdk import get_client

dotenv.load_dotenv()

# Get email from command line
parser = argparse.ArgumentParser()
parser.add_argument("email", help="Your email address for testing")
args = parser.parse_args()

# Create two test emails from the base email
base_email = args.email.split("@")
email1 = f"{base_email[0]}+1@{base_email[1]}"
email2 = f"{base_email[0]}+2@{base_email[1]}"

# Initialize auth provider settings
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]


async def login(email: str, password: str):
    """Get an access token for an existing user."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={
                "email": email,
                "password": password
            },
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json"
            },
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            raise ValueError(f"Login failed: {response.status_code} - {response.text}")


async def main():
    password = "secure-password"

    # Log in as user 1
    user1_token = await login(email1, password)
    user1_client = get_client(
        url="http://localhost:2024", headers={"Authorization": f"Bearer {user1_token}"}
    )

    # Create a thread as user 1
    thread = await user1_client.threads.create()
    print(f"✅ User 1 created thread: {thread['thread_id']}")

    # Try to access without a token
    unauthenticated_client = get_client(url="http://localhost:2024")
    try:
        await unauthenticated_client.threads.create()
        print("❌ Unauthenticated access should fail!")
    except Exception as e:
        print("✅ Unauthenticated access blocked:", e)

    # Try to access user 1's thread as user 2
    user2_token = await login(email2, password)
    user2_client = get_client(
        url="http://localhost:2024", headers={"Authorization": f"Bearer {user2_token}"}
    )

    try:
        await user2_client.threads.get(thread["thread_id"])
        print("❌ User 2 shouldn't see User 1's thread!")
    except Exception as e:
        print("✅ User 2 blocked from User 1's thread:", e)


if __name__ == "__main__":
    asyncio.run(main())
```

Fetch the SUPABASE_ANON_KEY that you copied from the Supabase dashboard in step (1), then run the test. Make sure the server is running (if you have run `langgraph dev`):

```bash
python test_oauth.py CHANGEME@example.com
```

> ➜  custom-auth SUPABASE_ANON_KEY=eyJh... python test_oauth.py CHANGEME@example.com
> ✅ User 1 created thread: d6af3754-95df-4176-aa10-dbd8dca40f1a
> ✅ Unauthenticated access blocked: Client error '403 Forbidden' for url 'http://localhost:2024/threads'
> ✅ User 2 blocked from User 1's thread: Client error '404 Not Found' for url 'http://localhost:2024/threads/d6af3754-95df-4176-aa10-dbd8dca40f1a'

Perfect! Our authentication and authorization are working together:
1. Users must log in to access the bot
2. Each user can only see their own threads

All our users are managed by the Supabase auth provider, so we don't need to implement any additional user management logic.

## Congratulations! 🎉

You've successfully built a production-ready authentication system for your LangGraph application! Let's review what you've accomplished:

1. Set up an authentication provider (Supabase in this case)
2. Added real user accounts with email/password authentication
3. Integrated JWT token validation into your LangGraph server
4. Implemented proper authorization to ensure users can only access their own data
5. Created a foundation that's ready to handle your next authentication challenge 🚀

This completes our authentication tutorial series. You now have the building blocks for a secure, production-ready LangGraph application.

## What's Next?

Now that you have production authentication, consider:

1. Building a web UI with your preferred framework (see the [Custom Auth](https://github.com/langchain-ai/custom-auth) template for an example)
2. Learn more about the other aspects of authentication and authorization in the [conceptual guide on authentication](../../concepts/auth.md).
3. Customize your handlers and setup further after reading the [reference docs](../../cloud/reference/sdk/python_sdk_ref.md#langgraph_sdk.auth.Auth).