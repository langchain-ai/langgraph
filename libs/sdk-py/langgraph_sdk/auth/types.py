"""Authentication and authorization types for LangGraph.

This module defines the core types used for authentication, authorization, and
request handling in LangGraph. It includes user protocols, authentication contexts,
and typed dictionaries for various API operations.

Note:
    All typing.TypedDict classes use total=False to make all fields typing.Optional by default.
"""

from __future__ import annotations

import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import typing_extensions

RunStatus = typing.Literal["pending", "error", "success", "timeout", "interrupted"]
"""Status of a run execution.

Values:
    - pending: Run is queued or in progress
    - error: Run failed with an error
    - success: Run completed successfully  
    - timeout: Run exceeded time limit
    - interrupted: Run was manually interrupted
"""

MultitaskStrategy = typing.Literal["reject", "rollback", "interrupt", "enqueue"]
"""Strategy for handling multiple concurrent tasks.

Values:
    - reject: Reject new tasks while one is in progress
    - rollback: Cancel current task and start new one
    - interrupt: Interrupt current task and start new one
    - enqueue: Queue new tasks to run after current one
"""

OnConflictBehavior = typing.Literal["raise", "do_nothing"]
"""Behavior when encountering conflicts.

Values:
    - raise: Raise an exception on conflict
    - do_nothing: Silently ignore conflicts
"""

IfNotExists = typing.Literal["create", "reject"]
"""Behavior when an entity doesn't exist.

Values:
    - create: Create the entity
    - reject: Reject the operation
"""

FilterType = (
    dict[
        str,
        str
        | dict[typing.Literal["$eq", "$contains"], str]
        | dict[typing.Literal["$contains"], list[str]],
    ]
    | dict[str, str]
)
"""Response type for authorization handlers.

Supports exact matches and operators:
    - Exact match shorthand: {"field": "value"}
    - Exact match: {"field": {"$eq": "value"}}
    - Contains (membership): {"field": {"$contains": "value"}}
    - Contains (subset containment): {"field": {"$contains": ["value1", "value2"]}}

Subset containment is only supported by newer versions of the LangGraph dev server;
install langgraph-runtime-inmem >= 0.14.1 to use this filter variant.

???+ example "Examples"

    Simple exact match filter for the resource owner:

    ```python
    filter = {"owner": "user-abcd123"}
    ```

    Explicit version of the exact match filter:

    ```python
    filter = {"owner": {"$eq": "user-abcd123"}}
    ```

    Containment (membership of a single element):

    ```python
    filter = {"participants": {"$contains": "user-abcd123"}}
    ```

    Containment (subset containment; all values must be present, but order doesn't matter):

    ```python
    filter = {"participants": {"$contains": ["user-abcd123", "user-efgh456"]}}
    ```

    Combining filters (treated as a logical `AND`):

    ```python
    filter = {"owner": "user-abcd123", "participants": {"$contains": "user-efgh456"}}
    ```
"""

ThreadStatus = typing.Literal["idle", "busy", "interrupted", "error"]
"""Status of a thread.

Values:
    - idle: Thread is available for work
    - busy: Thread is currently processing
    - interrupted: Thread was interrupted
    - error: Thread encountered an error
"""

MetadataInput = dict[str, typing.Any]
"""Type for arbitrary metadata attached to entities.

Allows storing custom key-value pairs with any entity.
Keys must be strings, values can be any JSON-serializable type.

???+ example "Examples"

    ```python
    metadata = {
        "created_by": "user123",
        "priority": 1,
        "tags": ["important", "urgent"]
    }
    ```
"""

HandlerResult = None | bool | FilterType
"""The result of a handler can be:
    * None | True: accept the request.
    * False: reject the request with a 403 error
    * FilterType: filter to apply
"""

Handler = Callable[..., Awaitable[HandlerResult]]

T = typing.TypeVar("T")


@typing.runtime_checkable
class MinimalUser(typing.Protocol):
    """User objects must at least expose the identity property."""

    @property
    def identity(self) -> str:
        """The unique identifier for the user.

        This could be a username, email, or any other unique identifier used
        to distinguish between different users in the system.
        """
        ...


class MinimalUserDict(typing.TypedDict, total=False):
    """The dictionary representation of a user."""

    identity: typing_extensions.Required[str]
    """The required unique identifier for the user."""
    display_name: str
    """The typing.Optional display name for the user."""
    is_authenticated: bool
    """Whether the user is authenticated. Defaults to True."""
    permissions: Sequence[str]
    """A list of permissions associated with the user.
    
    You can use these in your `@auth.on` authorization logic to determine
    access permissions to different resources.
    """


@typing.runtime_checkable
class BaseUser(typing.Protocol):
    """The base ASGI user protocol"""

    @property
    def is_authenticated(self) -> bool:
        """Whether the user is authenticated."""
        ...

    @property
    def display_name(self) -> str:
        """The display name of the user."""
        ...

    @property
    def identity(self) -> str:
        """The unique identifier for the user."""
        ...

    @property
    def permissions(self) -> Sequence[str]:
        """The permissions associated with the user."""
        ...

    def __getitem__(self, key):
        """Get a key from your minimal user dict."""
        ...

    def __contains__(self, key):
        """Check if a property exists."""
        ...

    def __iter__(self):
        """Iterate over the keys of the user."""
        ...


class StudioUser:
    """A user object that's populated from authenticated requests from the LangGraph studio.

    Note: Studio auth can be disabled in your `langgraph.json` config.

    ```json
    {
      "auth": {
        "disable_studio_auth": true
      }
    }
    ```

    You can use `isinstance` checks in your authorization handlers (`@auth.on`) to control access specifically
    for developers accessing the instance from the LangGraph Studio UI.

    ???+ example "Examples"

        ```python
        @auth.on
        async def allow_developers(ctx: Auth.types.AuthContext, value: Any) -> None:
            if isinstance(ctx.user, Auth.types.StudioUser):
                return None
            ...
            return False
        ```
    """

    __slots__ = ("_is_authenticated", "_permissions", "username")

    def __init__(self, username: str, is_authenticated: bool = False) -> None:
        self.username = username
        self._is_authenticated = is_authenticated
        self._permissions = ["authenticated"] if is_authenticated else []

    @property
    def is_authenticated(self) -> bool:
        return self._is_authenticated

    @property
    def display_name(self) -> str:
        return self.username

    @property
    def identity(self) -> str:
        return self.username

    @property
    def permissions(self) -> Sequence[str]:
        return self._permissions


Authenticator = Callable[
    ...,
    Awaitable[
        MinimalUser
        | str
        | BaseUser
        | MinimalUserDict
        | typing.Mapping[str, typing.Any],
    ],
]
"""Type for authentication functions.

An authenticator can return either:
1. A string (user_id)
2. A dict containing {"identity": str, "permissions": list[str]}
3. An object with identity and permissions properties

Permissions can be used downstream by your authorization logic to determine
access permissions to different resources.

The authenticate decorator will automatically inject any of the following parameters
by name if they are included in your function signature:

Parameters:
    request (Request): The raw ASGI request object
    body (dict): The parsed request body
    path (str): The request path
    method (str): The HTTP method (GET, POST, etc.)
    path_params (dict[str, str] | None): URL path parameters
    query_params (dict[str, str] | None): URL query parameters
    headers (dict[str, bytes] | None): Request headers
    authorization (str | None): The Authorization header value (e.g. "Bearer <token>")

???+ example "Examples"

    Basic authentication with token:

    ```python
    from langgraph_sdk import Auth

    auth = Auth()

    @auth.authenticate
    async def authenticate1(authorization: str) -> Auth.types.MinimalUserDict:
        return await get_user(authorization)
    ```

    Authentication with multiple parameters:

    ```    
    @auth.authenticate
    async def authenticate2(
        method: str,
        path: str,
        headers: dict[str, bytes]
    ) -> Auth.types.MinimalUserDict:
        # Custom auth logic using method, path and headers
        user = verify_request(method, path, headers)
        return user
    ```

    Accepting the raw ASGI request:

    ```python
    MY_SECRET = "my-secret-key"
    @auth.authenticate
    async def get_current_user(request: Request) -> Auth.types.MinimalUserDict:
        try:
            token = (request.headers.get("authorization") or "").split(" ", 1)[1]
            payload = jwt.decode(token, MY_SECRET, algorithms=["HS256"])
        except (IndexError, InvalidTokenError):
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.myauth-provider.com/auth/v1/user",
                headers={"Authorization": f"Bearer {MY_SECRET}"}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="User not found")
                
            user_data = response.json()
            return {
                "identity": user_data["id"],
                "display_name": user_data.get("name"),
                "permissions": user_data.get("permissions", []),
                "is_authenticated": True,
            }
    ```
"""


@dataclass(slots=True)
class BaseAuthContext:
    """Base class for authentication context.

    Provides the fundamental authentication information needed for
    authorization decisions.
    """

    permissions: Sequence[str]
    """The permissions granted to the authenticated user."""

    user: BaseUser
    """The authenticated user."""


@typing.final
@dataclass(slots=True)
class AuthContext(BaseAuthContext):
    """Complete authentication context with resource and action information.

    Extends BaseAuthContext with specific resource and action being accessed,
    allowing for fine-grained access control decisions.
    """

    resource: typing.Literal["runs", "threads", "crons", "assistants", "store"]
    """The resource being accessed."""

    action: typing.Literal[
        "create",
        "read",
        "update",
        "delete",
        "search",
        "create_run",
        "put",
        "get",
        "list_namespaces",
    ]
    """The action being performed on the resource.

    Most resources support the following actions:
    - create: Create a new resource
    - read: Read information about a resource
    - update: Update an existing resource
    - delete: Delete a resource
    - search: Search for resources

    The store supports the following actions:
    - put: Add or update an item in the store
    - get: Get an item from the store
    - search: Search for items within a namespace prefix
    - delete: Delete an item from the store
    - list_namespaces: List the namespaces in the store
    """


class ThreadTTL(typing.TypedDict, total=False):
    """Time-to-live configuration for a thread.

    Matches the OpenAPI schema where TTL is represented as an object with
    an optional strategy and a time value in minutes.
    """

    strategy: typing.Literal["delete"]
    """TTL strategy. Currently only 'delete' is supported."""

    ttl: int
    """Time-to-live in minutes from now until the thread should be swept."""


class ThreadsCreate(typing.TypedDict, total=False):
    """Parameters for creating a new thread.

    ???+ example "Examples"

        ```python
        create_params = {
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "metadata": {"owner": "user123"},
            "if_exists": "do_nothing"
        }
        ```
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    metadata: MetadataInput
    """typing.Optional metadata to attach to the thread."""

    if_exists: OnConflictBehavior
    """Behavior when a thread with the same ID already exists."""

    ttl: ThreadTTL
    """Optional TTL configuration for the thread."""


class ThreadsRead(typing.TypedDict, total=False):
    """Parameters for reading thread state or run information.

    This type is used in three contexts:
    1. Reading thread, thread version, or thread state information: Only thread_id is provided
    2. Reading run information: Both thread_id and run_id are provided
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    run_id: UUID | None
    """Run ID to filter by. Only used when reading run information within a thread."""


class ThreadsUpdate(typing.TypedDict, total=False):
    """Parameters for updating a thread or run.

    Called for updates to a thread, thread version, or run
    cancellation.
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    metadata: MetadataInput
    """typing.Optional metadata to update."""

    action: typing.Literal["interrupt", "rollback"] | None
    """typing.Optional action to perform on the thread."""


class ThreadsDelete(typing.TypedDict, total=False):
    """Parameters for deleting a thread.

    Called for deletes to a thread, thread version, or run
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    run_id: UUID | None
    """typing.Optional run ID to filter by."""


class ThreadsSearch(typing.TypedDict, total=False):
    """Parameters for searching threads.

    Called for searches to threads or runs.
    """

    metadata: MetadataInput
    """typing.Optional metadata to filter by."""

    values: MetadataInput
    """typing.Optional values to filter by."""

    status: ThreadStatus | None
    """typing.Optional status to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""

    ids: Sequence[UUID] | None
    """typing.Optional list of thread IDs to filter by."""

    thread_id: UUID | None
    """typing.Optional thread ID to filter by."""


class RunsCreate(typing.TypedDict, total=False):
    """Payload for creating a run.

    ???+ example "Examples"

        ```python
        create_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "run_id": UUID("123e4567-e89b-12d3-a456-426614174002"),
            "status": "pending",
            "metadata": {"owner": "user123"},
            "prevent_insert_if_inflight": True,
            "multitask_strategy": "reject",
            "if_not_exists": "create",
            "after_seconds": 10,
            "kwargs": {"key": "value"},
            "action": "interrupt"
        }
        ```
    """

    assistant_id: UUID | None
    """typing.Optional assistant ID to use for this run."""

    thread_id: UUID | None
    """typing.Optional thread ID to use for this run."""

    run_id: UUID | None
    """typing.Optional run ID to use for this run."""

    status: RunStatus | None
    """typing.Optional status for this run."""

    metadata: MetadataInput
    """typing.Optional metadata for the run."""

    prevent_insert_if_inflight: bool
    """Prevent inserting a new run if one is already in flight."""

    multitask_strategy: MultitaskStrategy
    """Multitask strategy for this run."""

    if_not_exists: IfNotExists
    """IfNotExists for this run."""

    after_seconds: int
    """Number of seconds to wait before creating the run."""

    kwargs: dict[str, typing.Any]
    """Keyword arguments to pass to the run."""

    action: typing.Literal["interrupt", "rollback"] | None
    """Action to take if updating an existing run."""


class AssistantsCreate(typing.TypedDict, total=False):
    """Payload for creating an assistant.

    ???+ example "Examples"

        ```python
        create_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"tags": ["tag1", "tag2"]},
            "context": {"key": "value"},
            "metadata": {"owner": "user123"},
            "if_exists": "do_nothing",
            "name": "Assistant 1"
        }
        ```
    """

    assistant_id: UUID
    """Unique identifier for the assistant."""

    graph_id: str
    """Graph ID to use for this assistant."""

    config: dict[str, typing.Any]
    """typing.Optional configuration for the assistant."""

    context: dict[str, typing.Any]

    metadata: MetadataInput
    """typing.Optional metadata to attach to the assistant."""

    if_exists: OnConflictBehavior
    """Behavior when an assistant with the same ID already exists."""

    name: str
    """Name of the assistant."""


class AssistantsRead(typing.TypedDict, total=False):
    """Payload for reading an assistant.

    ???+ example "Examples"

        ```python
        read_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "metadata": {"owner": "user123"}
        }
        ```
    """

    assistant_id: UUID
    """Unique identifier for the assistant."""

    metadata: MetadataInput
    """typing.Optional metadata to filter by."""


class AssistantsUpdate(typing.TypedDict, total=False):
    """Payload for updating an assistant.

    ???+ example "Examples"

        ```python
        update_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"tags": ["tag1", "tag2"]},
            "context": {"key": "value"},
            "metadata": {"owner": "user123"},
            "name": "Assistant 1",
            "version": 1
        }
        ```
    """

    assistant_id: UUID
    """Unique identifier for the assistant."""

    graph_id: str | None
    """typing.Optional graph ID to update."""

    config: dict[str, typing.Any]
    """typing.Optional configuration to update."""

    context: dict[str, typing.Any]
    """The static context of the assistant."""

    metadata: MetadataInput
    """typing.Optional metadata to update."""

    name: str | None
    """typing.Optional name to update."""

    version: int | None
    """typing.Optional version to update."""


class AssistantsDelete(typing.TypedDict):
    """Payload for deleting an assistant.

    ???+ example "Examples"

        ```python
        delete_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    assistant_id: UUID
    """Unique identifier for the assistant."""


class AssistantsSearch(typing.TypedDict):
    """Payload for searching assistants.

    ???+ example "Examples"

        ```python
        search_params = {
            "graph_id": "graph123",
            "metadata": {"owner": "user123"},
            "limit": 10,
            "offset": 0
        }
        ```
    """

    graph_id: str | None
    """typing.Optional graph ID to filter by."""

    metadata: MetadataInput
    """typing.Optional metadata to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""


class CronsCreate(typing.TypedDict, total=False):
    """Payload for creating a cron job.

    ???+ example "Examples"

        ```python
        create_params = {
            "payload": {"key": "value"},
            "schedule": "0 0 * * *",
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "user_id": "user123",
            "end_time": datetime(2024, 3, 16, 10, 0, 0)
        }
        ```
    """

    payload: dict[str, typing.Any]
    """Payload for the cron job."""

    schedule: str
    """Schedule for the cron job."""

    cron_id: UUID | None
    """typing.Optional unique identifier for the cron job."""

    thread_id: UUID | None
    """typing.Optional thread ID to use for this cron job."""

    user_id: str | None
    """typing.Optional user ID to use for this cron job."""

    end_time: datetime | None
    """typing.Optional end time for the cron job."""


class CronsDelete(typing.TypedDict):
    """Payload for deleting a cron job.

    ???+ example "Examples"

        ```python
        delete_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    cron_id: UUID
    """Unique identifier for the cron job."""


class CronsRead(typing.TypedDict):
    """Payload for reading a cron job.

    ???+ example "Examples"

        ```python
        read_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000")
        }
        ```
    """

    cron_id: UUID
    """Unique identifier for the cron job."""


class CronsUpdate(typing.TypedDict, total=False):
    """Payload for updating a cron job.

    ???+ example "Examples"

        ```python
        update_params = {
            "cron_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "payload": {"key": "value"},
            "schedule": "0 0 * * *"
        }
        ```
    """

    cron_id: UUID
    """Unique identifier for the cron job."""

    payload: dict[str, typing.Any] | None
    """typing.Optional payload to update."""

    schedule: str | None
    """typing.Optional schedule to update."""


class CronsSearch(typing.TypedDict, total=False):
    """Payload for searching cron jobs.

    ???+ example "Examples"

        ```python
        search_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "thread_id": UUID("123e4567-e89b-12d3-a456-426614174001"),
            "limit": 10,
            "offset": 0
        }
        ```
    """

    assistant_id: UUID | None
    """typing.Optional assistant ID to filter by."""

    thread_id: UUID | None
    """typing.Optional thread ID to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""


class StoreGet(typing.TypedDict):
    """Operation to retrieve a specific item by its namespace and key.

    This dict is mutable — auth handlers can modify `namespace` to enforce
    access scoping (e.g., prepending the user's identity).
    """

    namespace: tuple[str, ...]
    """Hierarchical path that uniquely identifies the item's location.

    Auth handlers can modify this to enforce per-user scoping.
    """

    key: str
    """Unique identifier for the item within its specific namespace."""


class StoreSearch(typing.TypedDict):
    """Operation to search for items within a specified namespace hierarchy.

    This dict is mutable — auth handlers can modify `namespace` to enforce
    access scoping (e.g., prepending the user's identity).
    """

    namespace: tuple[str, ...]
    """Prefix filter for defining the search scope.

    Auth handlers can modify this to enforce per-user scoping.
    """

    filter: dict[str, typing.Any] | None
    """Key-value pairs for filtering results based on exact matches or comparison operators."""

    limit: int
    """Maximum number of items to return in the search results."""

    offset: int
    """Number of matching items to skip for pagination."""

    query: str | None
    """Natural language search query for semantic search capabilities."""


class StoreListNamespaces(typing.TypedDict):
    """Operation to list and filter namespaces in the store.

    This dict is mutable — auth handlers can modify `namespace` (the prefix)
    to enforce access scoping (e.g., prepending the user's identity).
    """

    namespace: tuple[str, ...] | None
    """Prefix filter for namespaces. Can be `None` if no prefix was provided.

    Auth handlers can modify this to enforce per-user scoping. When `None`,
    handlers should set it to `(user_id,)` to scope listing to the user's namespaces.
    """

    suffix: tuple[str, ...] | None
    """Optional conditions for filtering namespaces."""

    max_depth: int | None
    """Maximum depth of namespace hierarchy to return.

    Note:
        Namespaces deeper than this level will be truncated.
    """

    limit: int
    """Maximum number of namespaces to return."""

    offset: int
    """Number of namespaces to skip for pagination."""


class StorePut(typing.TypedDict):
    """Operation to store, update, or delete an item in the store.

    This dict is mutable — auth handlers can modify `namespace` to enforce
    access scoping (e.g., prepending the user's identity).
    """

    namespace: tuple[str, ...]
    """Hierarchical path that identifies the location of the item.

    Auth handlers can modify this to enforce per-user scoping.
    """

    key: str
    """Unique identifier for the item within its namespace."""

    value: dict[str, typing.Any] | None
    """The data to store, or `None` to mark the item for deletion."""

    index: typing.Literal[False] | list[str] | None
    """Optional index configuration for full-text search."""


class StoreDelete(typing.TypedDict):
    """Operation to delete an item from the store.

    This dict is mutable — auth handlers can modify `namespace` to enforce
    access scoping (e.g., prepending the user's identity).
    """

    namespace: tuple[str, ...]
    """Hierarchical path that uniquely identifies the item's location.

    Auth handlers can modify this to enforce per-user scoping.
    """

    key: str
    """Unique identifier for the item within its specific namespace."""


class on:
    """Namespace for type definitions of different API operations.

    This class organizes type definitions for create, read, update, delete,
    and search operations across different resources (threads, assistants, crons).

    ???+ note "Usage"
        ```python
        from langgraph_sdk import Auth

        auth = Auth()

        @auth.on
        def handle_all(params: Auth.on.value):
            raise Exception("Not authorized")

        @auth.on.threads.create
        def handle_thread_create(params: Auth.on.threads.create.value):
            # Handle thread creation
            pass

        @auth.on.assistants.search
        def handle_assistant_search(params: Auth.on.assistants.search.value):
            # Handle assistant search
            pass
        ```
    """

    value = dict[str, typing.Any]

    class threads:
        """Types for thread-related operations."""

        value = (
            ThreadsCreate | ThreadsRead | ThreadsUpdate | ThreadsDelete | ThreadsSearch
        )

        class create:
            """Type for thread creation parameters."""

            value = ThreadsCreate

        class create_run:
            """Type for creating or streaming a run."""

            value = RunsCreate

        class read:
            """Type for thread read parameters."""

            value = ThreadsRead

        class update:
            """Type for thread update parameters."""

            value = ThreadsUpdate

        class delete:
            """Type for thread deletion parameters."""

            value = ThreadsDelete

        class search:
            """Type for thread search parameters."""

            value = ThreadsSearch

    class assistants:
        """Types for assistant-related operations."""

        value = (
            AssistantsCreate
            | AssistantsRead
            | AssistantsUpdate
            | AssistantsDelete
            | AssistantsSearch
        )

        class create:
            """Type for assistant creation parameters."""

            value = AssistantsCreate

        class read:
            """Type for assistant read parameters."""

            value = AssistantsRead

        class update:
            """Type for assistant update parameters."""

            value = AssistantsUpdate

        class delete:
            """Type for assistant deletion parameters."""

            value = AssistantsDelete

        class search:
            """Type for assistant search parameters."""

            value = AssistantsSearch

    class crons:
        """Types for cron-related operations."""

        value = CronsCreate | CronsRead | CronsUpdate | CronsDelete | CronsSearch

        class create:
            """Type for cron creation parameters."""

            value = CronsCreate

        class read:
            """Type for cron read parameters."""

            value = CronsRead

        class update:
            """Type for cron update parameters."""

            value = CronsUpdate

        class delete:
            """Type for cron deletion parameters."""

            value = CronsDelete

        class search:
            """Type for cron search parameters."""

            value = CronsSearch

    class store:
        """Types for store-related operations."""

        value = StoreGet | StoreSearch | StoreListNamespaces | StorePut | StoreDelete

        class put:
            """Type for store put parameters."""

            value = StorePut

        class get:
            """Type for store get parameters."""

            value = StoreGet

        class search:
            """Type for store search parameters."""

            value = StoreSearch

        class delete:
            """Type for store delete parameters."""

            value = StoreDelete

        class list_namespaces:
            """Type for store list namespaces parameters."""

            value = StoreListNamespaces


__all__ = [
    "AssistantsCreate",
    "AssistantsDelete",
    "AssistantsRead",
    "AssistantsSearch",
    "AssistantsUpdate",
    "MetadataInput",
    "RunsCreate",
    "StoreDelete",
    "StoreGet",
    "StoreListNamespaces",
    "StorePut",
    "StoreSearch",
    "ThreadsCreate",
    "ThreadsDelete",
    "ThreadsRead",
    "ThreadsSearch",
    "ThreadsUpdate",
    "on",
]
