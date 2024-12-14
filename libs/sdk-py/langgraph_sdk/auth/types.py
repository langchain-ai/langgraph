"""Authentication and authorization types for LangGraph.

This module defines the core types used for authentication, authorization, and
request handling in LangGraph. It includes user protocols, authentication contexts,
and typed dictionaries for various API operations.

Note:
    All TypedDict classes use total=False to make all fields optional by default.
"""

import functools
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    final,
    runtime_checkable,
)
from uuid import UUID

RunStatus = Literal["pending", "error", "success", "timeout", "interrupted"]
"""Status of a run execution.

Values:
    - pending: Run is queued or in progress
    - error: Run failed with an error
    - success: Run completed successfully  
    - timeout: Run exceeded time limit
    - interrupted: Run was manually interrupted
"""

MultitaskStrategy = Literal["reject", "rollback", "interrupt", "enqueue"]
"""Strategy for handling multiple concurrent tasks.

Values:
    - reject: Reject new tasks while one is in progress
    - rollback: Cancel current task and start new one
    - interrupt: Interrupt current task and start new one
    - enqueue: Queue new tasks to run after current one
"""

OnConflictBehavior = Literal["raise", "do_nothing"]
"""Behavior when encountering conflicts.

Values:
    - raise: Raise an exception on conflict
    - do_nothing: Silently ignore conflicts
"""

IfNotExists = Literal["create", "reject"]
"""Behavior when an entity doesn't exist.

Values:
    - create: Create the entity
    - reject: Reject the operation
"""

FilterType = Union[
    Dict[str, Union[str, Dict[Literal["$eq", "$contains"], str]]], Dict[str, str]
]
"""Type for filtering queries.

Supports exact matches and operators:
    - Simple match: {"field": "value"}
    - Equals: {"field": {"$eq": "value"}}
    - Contains: {"field": {"$contains": "value"}}

???+ example "Examples"
    ```python
    # Simple match
    filter = {"status": "pending"}
    
    # Equals operator
    filter = {"status": {"$eq": "success"}}
    
    # Contains operator
    filter = {"metadata.tags": {"$contains": "important"}}
    ```
"""

ThreadStatus = Literal["idle", "busy", "interrupted", "error"]
"""Status of a thread.

Values:
    - idle: Thread is available for work
    - busy: Thread is currently processing
    - interrupted: Thread was interrupted
    - error: Thread encountered an error
"""

MetadataInput = Dict[str, Any]
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

HandlerResult = Union[None, bool, FilterType]
"""The result of a handler can be:
- None | True: accept the request.
- False: reject the request with a 403 error
- FilterType: filter to apply
"""

Handler = Callable[..., Awaitable[HandlerResult]]

T = TypeVar("T")


def _slotify(fn: T) -> T:
    if sys.version_info >= (3, 10):  # noqa: UP036
        return functools.partial(fn, slots=True)  # type: ignore
    return fn


dataclass = _slotify(dataclass)


@runtime_checkable
class MinimalUser(Protocol):
    """User objects must at least expose the identity property."""

    @property
    def identity(self) -> str:
        """The unique identifier for the user.

        This could be a username, email, or any other unique identifier used
        to distinguish between different users in the system.
        """
        ...


class MinimalUserDict(TypedDict, total=False):
    """The minimal user dictionary."""

    identity: str
    display_name: str
    is_authenticated: bool


@runtime_checkable
class BaseUser(Protocol):
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


Authenticator = Callable[
    ..., Awaitable[tuple[list[str], Union[MinimalUser, str, MinimalUserDict]]]
]
"""Type for authentication functions.

An authenticator can return either:
1. A tuple of (scopes, MinimalUser/BaseUser)
2. A tuple of (scopes, str) where str is the user identity

Scopes can be used downstream by your authorization logic to determine
access permissions to different resources.

The authenticate decorator will automatically inject any of the following parameters
by name if they are included in your function signature:

Parameters:
    request (Request): The raw ASGI request object
    body (dict): The parsed request body
    path (str): The request path
    method (str): The HTTP method (GET, POST, etc.)
    scopes (list[str]): The required scopes for this endpoint
    path_params (dict[str, str] | None): URL path parameters
    query_params (dict[str, str] | None): URL query parameters
    headers (dict[str, bytes] | None): Request headers
    authorization (str | None): The Authorization header value

???+ example "Examples"
    Basic authentication with token:
    ```python
    from langgraph_sdk import Auth

    auth = Auth()

    @auth.authenticate
    async def authenticate1(authorization: str) -> tuple[list[str], MinimalUser]:
        user = await get_user(authorization)
        return ["read", "write"], user
    ```

    Authentication with multiple parameters:
    ```    
    @auth.authenticate
    async def authenticate2(
        method: str,
        path: str,
        headers: dict[str, bytes]
    ) -> tuple[list[str], str]:
        # Custom auth logic using method, path and headers
        user_id = verify_request(method, path, headers)
        return ["read"], user_id
    ```

    Accepting the raw ASGI request:
    ```python
    MY_SECRET = "my-secret-key"
    @auth.authenticate
    async def get_current_user(request: Request) -> tuple[list[str], dict]:
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
            return payload.get("role", []), {
                "username": user_data["id"],
                "email": user_data["email"],
                "full_name": user_data.get("user_metadata", {}).get("full_name")
            }
    ```
"""


@dataclass
class BaseAuthContext:
    """Base class for authentication context.

    Provides the fundamental authentication information needed for
    authorization decisions.
    """

    scopes: Sequence[str]
    """The scopes granted to the authenticated user."""

    user: BaseUser
    """The authenticated user."""


@final
@dataclass
class AuthContext(BaseAuthContext):
    """Complete authentication context with resource and action information.

    Extends BaseAuthContext with specific resource and action being accessed,
    allowing for fine-grained access control decisions.
    """

    resource: Literal["runs", "threads", "crons", "assistants"]
    """The resource being accessed."""

    action: Literal["create", "read", "update", "delete", "search", "create_run"]
    """The action being performed on the resource."""


class ThreadsCreate(TypedDict, total=False):
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
    """Optional metadata to attach to the thread."""

    if_exists: OnConflictBehavior
    """Behavior when a thread with the same ID already exists."""


class ThreadsRead(TypedDict, total=False):
    """Parameters for reading thread state or run information.

    This type is used in three contexts:
    1. Reading thread, thread version, or thread state information: Only thread_id is provided
    2. Reading run information: Both thread_id and run_id are provided
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    run_id: Optional[UUID]
    """Run ID to filter by. Only used when reading run information within a thread."""


class ThreadsUpdate(TypedDict, total=False):
    """Parameters for updating a thread or run.

    Called for updates to a thread, thread version, or run
    cancellation.
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    metadata: MetadataInput
    """Optional metadata to update."""

    action: Optional[Literal["interrupt", "rollback"]]
    """Optional action to perform on the thread."""


class ThreadsDelete(TypedDict, total=False):
    """Parameters for deleting a thread.

    Called for deletes to a thread, thread version, or run
    """

    thread_id: UUID
    """Unique identifier for the thread."""

    run_id: Optional[UUID]
    """Optional run ID to filter by."""


class ThreadsSearch(TypedDict, total=False):
    """Parameters for searching threads.

    Called for searches to threads or runs.
    """

    metadata: MetadataInput
    """Optional metadata to filter by."""

    values: MetadataInput
    """Optional values to filter by."""

    status: Optional[ThreadStatus]
    """Optional status to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""

    thread_id: Optional[UUID]
    """Optional thread ID to filter by."""


class RunsCreate(TypedDict, total=False):
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

    assistant_id: Optional[UUID]
    """Optional assistant ID to use for this run."""

    thread_id: Optional[UUID]
    """Optional thread ID to use for this run."""

    run_id: Optional[UUID]
    """Optional run ID to use for this run."""

    status: Optional[RunStatus]
    """Optional status for this run."""

    metadata: MetadataInput
    """Optional metadata for the run."""

    prevent_insert_if_inflight: bool
    """Prevent inserting a new run if one is already in flight."""

    multitask_strategy: MultitaskStrategy
    """Multitask strategy for this run."""

    if_not_exists: IfNotExists
    """IfNotExists for this run."""

    after_seconds: int
    """Number of seconds to wait before creating the run."""

    kwargs: Dict[str, Any]
    """Keyword arguments to pass to the run."""

    action: Optional[Literal["interrupt", "rollback"]]
    """Action to take if updating an existing run."""


class AssistantsCreate(TypedDict, total=False):
    """Payload for creating an assistant.

    ???+ example "Examples"
        ```python
        create_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"key": "value"},
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

    config: Optional[Union[Dict[str, Any], Any]]
    """Optional configuration for the assistant."""

    metadata: MetadataInput
    """Optional metadata to attach to the assistant."""

    if_exists: OnConflictBehavior
    """Behavior when an assistant with the same ID already exists."""

    name: str
    """Name of the assistant."""


class AssistantsRead(TypedDict, total=False):
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
    """Optional metadata to filter by."""


class AssistantsUpdate(TypedDict, total=False):
    """Payload for updating an assistant.

    ???+ example "Examples"
        ```python
        update_params = {
            "assistant_id": UUID("123e4567-e89b-12d3-a456-426614174000"),
            "graph_id": "graph123",
            "config": {"key": "value"},
            "metadata": {"owner": "user123"},
            "name": "Assistant 1",
            "version": 1
        }
        ```
    """

    assistant_id: UUID
    """Unique identifier for the assistant."""

    graph_id: Optional[str]
    """Optional graph ID to update."""

    config: Optional[Union[Dict[str, Any], Any]]
    """Optional configuration to update."""

    metadata: MetadataInput
    """Optional metadata to update."""

    name: Optional[str]
    """Optional name to update."""

    version: Optional[int]
    """Optional version to update."""


class AssistantsDelete(TypedDict):
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


class AssistantsSearch(TypedDict):
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

    graph_id: Optional[str]
    """Optional graph ID to filter by."""

    metadata: MetadataInput
    """Optional metadata to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""


class CronsCreate(TypedDict, total=False):
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

    payload: Dict[str, Any]
    """Payload for the cron job."""

    schedule: str
    """Schedule for the cron job."""

    cron_id: Optional[UUID]
    """Optional unique identifier for the cron job."""

    thread_id: Optional[UUID]
    """Optional thread ID to use for this cron job."""

    user_id: Optional[str]
    """Optional user ID to use for this cron job."""

    end_time: Optional[datetime]
    """Optional end time for the cron job."""


class CronsDelete(TypedDict):
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


class CronsRead(TypedDict):
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


class CronsUpdate(TypedDict, total=False):
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

    payload: Optional[Dict[str, Any]]
    """Optional payload to update."""

    schedule: Optional[str]
    """Optional schedule to update."""


class CronsSearch(TypedDict, total=False):
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

    assistant_id: Optional[UUID]
    """Optional assistant ID to filter by."""

    thread_id: Optional[UUID]
    """Optional thread ID to filter by."""

    limit: int
    """Maximum number of results to return."""

    offset: int
    """Offset for pagination."""


class on:
    """Namespace for type definitions of different API operations.

    This class organizes type definitions for create, read, update, delete,
    and search operations across different resources (threads, assistants, crons).

    ???+ note "Usage"
        ```python
        from langgraph_sdk import Auth

        @Auth.on
        def handle_all(params: Auth.on.value):
            raise Exception("Not authorized")

        @Auth.on.threads.create
        def handle_thread_create(params: Auth.on.threads.create.value):
            # Handle thread creation
            pass

        @Auth.on.assistants.search
        def handle_assistant_search(params: Auth.on.assistants.search.value):
            # Handle assistant search
            pass
        ```
    """

    value = Dict[str, Any]

    class threads:
        """Types for thread-related operations."""

        value = Union[
            ThreadsCreate, ThreadsRead, ThreadsUpdate, ThreadsDelete, ThreadsSearch
        ]

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

        value = Union[
            AssistantsCreate,
            AssistantsRead,
            AssistantsUpdate,
            AssistantsDelete,
            AssistantsSearch,
        ]

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

        value = Union[CronsCreate, CronsRead, CronsUpdate, CronsDelete, CronsSearch]

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


__all__ = [
    "on",
    "MetadataInput",
    "RunsCreate",
    "ThreadsCreate",
    "ThreadsRead",
    "ThreadsUpdate",
    "ThreadsDelete",
    "ThreadsSearch",
    "AssistantsCreate",
    "AssistantsRead",
    "AssistantsUpdate",
    "AssistantsDelete",
    "AssistantsSearch",
]
