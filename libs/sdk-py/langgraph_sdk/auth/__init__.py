from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

from langgraph_sdk.auth import types

TH = TypeVar("TH", bound=types.Handler)
AH = TypeVar("AH", bound=types.Authenticator)


class Auth:
    """Authentication and authorization management for LangGraph.

    The Auth class provides a unified system for handling authentication and
    authorization in LangGraph applications. It supports:

    1. Authentication via a decorator-based handler system
    2. Fine-grained authorization rules for different resources and actions
    3. Global and resource-specific authorization handlers

    ???+ example "Basic Usage"
        ```python
        from langgraph_sdk import Auth

        auth = Auth()

        @auth.authenticate
        async def authenticate(authorization: str) -> tuple[list[str], str]:
            # Verify token and return (scopes, user_id)
            user_id = verify_token(authorization)
            return ["read", "write"], user_id

        # Global fallback handler
        @auth.on
        async def authorize_default(params: Auth.on.value):
            return False # Reject all requests (default behavior)

        @auth.on.threads.create
        async def authorize_thread_create(params: Auth.on.threads.create.value):
            # Allow the allowed user to create a thread
            assert params.get("metadata", {}).get("owner") == "allowed_user"
        ```

    ???+ note "Request Processing Flow"
        1. Authentication is performed first on every request
        2. For authorization, the most specific matching handler is called:
           - If a handler exists for the exact resource and action, it is used
           - Otherwise, if a handler exists for the resource with any action, it is used
           - Finally, if no specific handlers match, the global handler is used (if any)

        This allows you to set default behavior with a global handler while
        overriding specific routes as needed.
    """

    __slots__ = (
        "on",
        "_handlers",
        "_global_handlers",
        "_authenticate_handler",
        "_handler_cache",
    )
    types = types
    """Reference to auth type definitions.
    
    Provides access to all type definitions used in the auth system,
    like ThreadsCreate, AssistantsRead, etc."""

    def __init__(self) -> None:
        self.on = _On(self)
        # These are accessed by the API. Changes to their names or types is
        # will be considered a breaking change.
        self._handlers: dict[tuple[str, str], list[types.Handler]] = {}
        self._global_handlers: list[types.Handler] = []
        self._authenticate_handler: Optional[types.Authenticator] = None
        self._handler_cache: dict[tuple[str, str], types.Handler] = {}

    def authenticate(self, fn: AH) -> AH:
        """Register an authentication handler function.

        The authentication handler is responsible for verifying credentials
        and returning user scopes. It can accept any of the following parameters
        by name:
            - request (Request): The raw ASGI request object
            - body (dict): The parsed request body
            - path (str): The request path
            - method (str): The HTTP method
            - scopes (list[str]): Required scopes
            - path_params (dict[str, str]): URL path parameters
            - query_params (dict[str, str]): URL query parameters
            - headers (dict[str, bytes]): Request headers
            - authorization (str): The Authorization header value

        Args:
            fn (Callable): The authentication handler function to register.
                Must return tuple[scopes, user]
                where scopes is a list of string claims (like "runs:read", etc.)
                and user is either a user object (or similar dict) or a user id string.

        Returns:
            The registered handler function.

        Raises:
            ValueError: If an authentication handler is already registered.

        ???+ example "Examples"
            Basic token authentication:
            ```python
            @auth.authenticate
            async def authenticate(authorization: str) -> tuple[list[str], str]:
                user_id = verify_token(authorization)
                return ["read"], user_id
            ```

            Complex authentication with request context:
            ```python
            @auth.authenticate
            async def authenticate(
                method: str,
                path: str,
                headers: dict[str, bytes]
            ) -> tuple[list[str], MinimalUser]:
                user = await verify_request(method, path, headers)
                return user.scopes, user
            ```
        """
        if self._authenticate_handler is not None:
            raise ValueError(
                "Authentication handler already set as {self._authenticate_handler}."
            )
        self._authenticate_handler = fn
        return fn


## Helper types & utilities

V = TypeVar("V", contravariant=True)


class _ActionHandler(Protocol[V]):
    async def __call__(
        self, *, ctx: types.AuthContext, value: V
    ) -> types.HandlerResult: ...


T = TypeVar("T", covariant=True)


class _ResourceActionOn(Generic[T]):
    def __init__(
        self,
        auth: Auth,
        resource: Literal["threads", "crons", "assistants"],
        action: Literal["create", "read", "update", "delete", "search", "create_run"],
        value: type[T],
    ) -> None:
        self.auth = auth
        self.resource = resource
        self.action = action
        self.value = value

    def __call__(self, fn: _ActionHandler[T]) -> _ActionHandler[T]:
        _validate_handler(fn)
        _register_handler(self.auth, self.resource, self.action, fn)
        return fn


VCreate = TypeVar("VCreate", covariant=True)
VUpdate = TypeVar("VUpdate", covariant=True)
VRead = TypeVar("VRead", covariant=True)
VDelete = TypeVar("VDelete", covariant=True)
VSearch = TypeVar("VSearch", covariant=True)


class _ResourceOn(Generic[VCreate, VRead, VUpdate, VDelete, VSearch]):
    """
    Generic base class for resource-specific handlers.
    """

    value: type[Union[VCreate, VUpdate, VRead, VDelete, VSearch]]

    Create: type[VCreate]
    Read: type[VRead]
    Update: type[VUpdate]
    Delete: type[VDelete]
    Search: type[VSearch]

    def __init__(
        self,
        auth: Auth,
        resource: Literal["threads", "crons", "assistants"],
    ) -> None:
        self.auth = auth
        self.resource = resource
        self.create: _ResourceActionOn[VCreate] = _ResourceActionOn(
            auth, resource, "create", self.Create
        )
        self.read: _ResourceActionOn[VRead] = _ResourceActionOn(
            auth, resource, "read", self.Read
        )
        self.update: _ResourceActionOn[VUpdate] = _ResourceActionOn(
            auth, resource, "update", self.Update
        )
        self.delete: _ResourceActionOn[VDelete] = _ResourceActionOn(
            auth, resource, "delete", self.Delete
        )
        self.search: _ResourceActionOn[VSearch] = _ResourceActionOn(
            auth, resource, "search", self.Search
        )

    @overload
    def __call__(
        self,
        fn: Union[
            _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
            _ActionHandler[dict[str, Any]],
        ],
    ) -> _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]]: ...

    @overload
    def __call__(
        self,
        *,
        resources: Union[str, Sequence[str]],
        actions: Optional[Union[str, Sequence[str]]] = None,
    ) -> Callable[
        [_ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]]],
        _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
    ]: ...

    def __call__(
        self,
        fn: Union[
            _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
            _ActionHandler[dict[str, Any]],
            None,
        ] = None,
        *,
        resources: Union[str, Sequence[str], None] = None,
        actions: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[
        _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
        Callable[
            [_ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]]],
            _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
        ],
    ]:
        if fn is not None:
            _validate_handler(fn)
            return cast(
                _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
                _register_handler(self.auth, self.resource, "*", fn),
            )

        def decorator(
            handler: _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
        ) -> _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]]:
            _validate_handler(handler)
            return cast(
                _ActionHandler[Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
                _register_handler(self.auth, self.resource, "*", handler),
            )

        return decorator


class _AssistantsOn(
    _ResourceOn[
        types.AssistantsCreate,
        types.AssistantsRead,
        types.AssistantsUpdate,
        types.AssistantsDelete,
        types.AssistantsSearch,
    ]
):
    value = Union[
        types.AssistantsCreate,
        types.AssistantsRead,
        types.AssistantsUpdate,
        types.AssistantsDelete,
        types.AssistantsSearch,
    ]
    Create = types.AssistantsCreate
    Read = types.AssistantsRead
    Update = types.AssistantsUpdate
    Delete = types.AssistantsDelete
    Search = types.AssistantsSearch


class _ThreadsOn(
    _ResourceOn[
        types.ThreadsCreate,
        types.ThreadsRead,
        types.ThreadsUpdate,
        types.ThreadsDelete,
        types.ThreadsSearch,
    ]
):
    value = Union[
        type[types.ThreadsCreate],
        type[types.ThreadsRead],
        type[types.ThreadsUpdate],
        type[types.ThreadsDelete],
        type[types.ThreadsSearch],
        type[types.RunsCreate],
    ]
    Create = types.ThreadsCreate
    Read = types.ThreadsRead
    Update = types.ThreadsUpdate
    Delete = types.ThreadsDelete
    Search = types.ThreadsSearch
    CreateRun = types.RunsCreate

    def __init__(
        self,
        auth: Auth,
        resource: Literal["threads", "crons", "assistants"],
    ) -> None:
        super().__init__(auth, resource)
        self.create_run: _ResourceActionOn[types.RunsCreate] = _ResourceActionOn(
            auth, resource, "create_run", self.CreateRun
        )


class _CronsOn(
    _ResourceOn[
        types.CronsCreate,
        types.CronsRead,
        types.CronsUpdate,
        types.CronsDelete,
        types.CronsSearch,
    ]
):
    value = type[
        Union[
            types.CronsCreate,
            types.CronsRead,
            types.CronsUpdate,
            types.CronsDelete,
            types.CronsSearch,
        ]
    ]

    Create = types.CronsCreate
    Read = types.CronsRead
    Update = types.CronsUpdate
    Delete = types.CronsDelete
    Search = types.CronsSearch


AHO = TypeVar("AHO", bound=_ActionHandler[dict[str, Any]])


class _On:
    """
    Entry point for @auth.on decorators.
    Provides access to specific resources."""

    __slots__ = (
        "_auth",
        "assistants",
        "threads",
        "runs",
        "crons",
        "value",
    )

    def __init__(self, auth: Auth) -> None:
        self._auth = auth
        self.assistants = _AssistantsOn(auth, "assistants")
        self.threads = _ThreadsOn(auth, "threads")
        self.crons = _CronsOn(auth, "crons")
        self.value = dict[str, Any]

    @overload
    def __call__(
        self,
        *,
        resources: Union[str, Sequence[str]],
        actions: Optional[Union[str, Sequence[str]]] = None,
    ) -> Callable[[AHO], AHO]: ...

    @overload
    def __call__(self, fn: AHO) -> AHO: ...

    def __call__(
        self,
        fn: Optional[AHO] = None,
        *,
        resources: Union[str, Sequence[str], None] = None,
        actions: Optional[Union[str, Sequence[str]]] = None,
    ) -> Union[AHO, Callable[[AHO], AHO]]:
        """Register a handler for specific resources and actions.

        Can be used as a decorator or with explicit resource/action parameters:

        @auth.on
        async def handler(): ...  # Global handler

        @auth.on(resources="threads")
        async def handler(): ...  # types.Handler for all thread actions

        @auth.on(resources="threads", actions="create")
        async def handler(): ...  # types.Handler for thread creation
        """
        if fn is not None:
            # Used as a plain decorator
            _register_handler(self._auth, None, None, fn)
            return fn

        # Used with parameters, return a decorator
        def decorator(handler: AHO) -> AHO:
            if isinstance(resources, str):
                resource_list = [resources]
            else:
                resource_list = list(resources) if resources is not None else ["*"]

            if isinstance(actions, str):
                action_list = [actions]
            else:
                action_list = list(actions) if actions is not None else ["*"]
            for resource in resource_list:
                for action in action_list:
                    _register_handler(self._auth, resource, action, handler)
            return handler

        return decorator


def _register_handler(
    auth: Auth, resource: Optional[str], action: Optional[str], fn: types.Handler
) -> types.Handler:
    _validate_handler(fn)
    resource = resource or "*"
    action = action or "*"
    if resource == "*" and action == "*":
        if auth._global_handlers:
            raise ValueError("Global handler already set.")
        auth._global_handlers.append(fn)
    else:
        r = resource if resource is not None else "*"
        a = action if action is not None else "*"
        if (r, a) in auth._handlers:
            raise ValueError(f"types.Handler already set for {r}, {a}.")
        auth._handlers[(r, a)] = [fn]
    return fn


def _validate_handler(fn: Callable[..., Any]) -> None:
    """Validates that an auth handler function meets the required signature.

    Auth handlers must:
    1. Be async functions
    2. Accept a ctx parameter of type AuthContext
    3. Accept a value parameter for the data being authorized
    """
    if not inspect.iscoroutinefunction(fn):
        raise ValueError(
            f"Auth handler '{fn.__name__}' must be an async function. "
            "Add 'async' before 'def' to make it asynchronous and ensure"
            " any IO operations are non-blocking."
        )

    sig = inspect.signature(fn)
    if "ctx" not in sig.parameters:
        raise ValueError(
            f"Auth handler '{fn.__name__}' must have a 'ctx: AuthContext' parameter. "
            "Update the function signature to include this required parameter."
        )
    if "value" not in sig.parameters:
        raise ValueError(
            f"Auth handler '{fn.__name__}' must have a 'value' parameter. "
            " The value contains the mutable data being sent to the endpoint."
            "Update the function signature to include this required parameter."
        )


__all__ = ["Auth", "types"]
