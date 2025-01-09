from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Sequence

from langgraph_sdk.auth import exceptions, types

TH = typing.TypeVar("TH", bound=types.Handler)
AH = typing.TypeVar("AH", bound=types.Authenticator)


class Auth:
    """Add custom authentication and authorization management to your LangGraph application.

    The Auth class provides a unified system for handling authentication and
    authorization in LangGraph applications. It supports custom user authentication
    protocols and fine-grained authorization rules for different resources and
    actions.

    To use, create a separate python file and add the path to the file to your
    LangGraph API configuration file (`langgraph.json`). Within that file, create
    an instance of the Auth class and register authentication and authorization
    handlers as needed.

    Example `langgraph.json` file:

    ```json
    {
      "dependencies": ["."],
      "graphs": {
        "agent": "./my_agent/agent.py:graph"
      },
      "env": ".env",
      "auth": {
        "path": "./auth.py:my_auth"
      }
    ```

    Then the LangGraph server will load your auth file and run it server-side whenever a request comes in.

    ???+ example "Basic Usage"
        ```python
        from langgraph_sdk import Auth

        my_auth = Auth()

        async def verify_token(token: str) -> str:
            # Verify token and return user_id
            # This would typically be a call to your auth server
            return "user_id"

        @auth.authenticate
        async def authenticate(authorization: str) -> str:
            # Verify token and return user_id
            result = await verify_token(authorization)
            if result != "user_id":
                raise Auth.exceptions.HTTPException(
                    status_code=401, detail="Unauthorized"
                )
            return result

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
        1. Authentication (your `@auth.authenticate` handler) is performed first on **every request**
        2. For authorization, the most specific matching handler is called:
            * If a handler exists for the exact resource and action, it is used (e.g., `@auth.on.threads.create`)
            * Otherwise, if a handler exists for the resource with any action, it is used (e.g., `@auth.on.threads`)
            * Finally, if no specific handlers match, the global handler is used (e.g., `@auth.on`)
            * If no global handler is set, the request is accepted

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

    exceptions = exceptions
    """Reference to auth exception definitions.
    
    Provides access to all exception definitions used in the auth system,
    like HTTPException, etc.    
    """

    def __init__(self) -> None:
        self.on = _On(self)
        """Entry point for authorization handlers that control access to specific resources.

        The on class provides a flexible way to define authorization rules for different
        resources and actions in your application. It supports three main usage patterns:

        1. Global handlers that run for all resources and actions
        2. Resource-specific handlers that run for all actions on a resource
        3. Resource and action specific handlers for fine-grained control

        Each handler must be an async function that accepts two parameters:
            - ctx (AuthContext): Contains request context and authenticated user info
            - value: The data being authorized (type varies by endpoint)

        The handler should return one of:

            - None or True: Accept the request
            - False: Reject with 403 error
            - FilterType: Apply filtering rules to the response
        
        ???+ example "Examples"
            Global handler for all requests:
            ```python
            @auth.on
            async def reject_unhandled_requests(ctx: AuthContext, value: Any) -> None:
                print(f"Request to {ctx.path} by {ctx.user.identity}")
                return False
            ```

            Resource-specific handler. This would take precedence over the global handler
            for all actions on the `threads` resource:
            ```python
            @auth.on.threads
            async def check_thread_access(ctx: AuthContext, value: Any) -> bool:
                # Allow access only to threads created by the user
                return value.get("created_by") == ctx.user.identity
            ```

            Resource and action specific handler:
            ```python
            @auth.on.threads.delete
            async def prevent_thread_deletion(ctx: AuthContext, value: Any) -> bool:
                # Only admins can delete threads
                return "admin" in ctx.user.permissions
            ```

            Multiple resources or actions:
            ```python
            @auth.on(resources=["threads", "runs"], actions=["create", "update"])
            async def rate_limit_writes(ctx: AuthContext, value: Any) -> bool:
                # Implement rate limiting for write operations
                return await check_rate_limit(ctx.user.identity)
            ```
        """
        # These are accessed by the API. Changes to their names or types is
        # will be considered a breaking change.
        self._handlers: dict[tuple[str, str], list[types.Handler]] = {}
        self._global_handlers: list[types.Handler] = []
        self._authenticate_handler: typing.Optional[types.Authenticator] = None
        self._handler_cache: dict[tuple[str, str], types.Handler] = {}

    def authenticate(self, fn: AH) -> AH:
        """Register an authentication handler function.

        The authentication handler is responsible for verifying credentials
        and returning user scopes. It can accept any of the following parameters
        by name:

            - request (Request): The raw ASGI request object
            - body (dict): The parsed request body
            - path (str): The request path, e.g., "/threads/abcd-1234-abcd-1234/runs/abcd-1234-abcd-1234/stream"
            - method (str): The HTTP method, e.g., "GET"
            - path_params (dict[str, str]): URL path parameters, e.g., {"thread_id": "abcd-1234-abcd-1234", "run_id": "abcd-1234-abcd-1234"}
            - query_params (dict[str, str]): URL query parameters, e.g., {"stream": "true"}
            - headers (dict[bytes, bytes]): Request headers
            - authorization (str | None): The Authorization header value (e.g., "Bearer <token>")

        Args:
            fn (Callable): The authentication handler function to register.
                Must return a representation of the user. This could be a:
                    - string (the user id)
                    - dict containing {"identity": str, "permissions": list[str]}
                    - or an object with identity and permissions properties
                Permissions can be optionally used by your handlers downstream.

        Returns:
            The registered handler function.

        Raises:
            ValueError: If an authentication handler is already registered.

        ???+ example "Examples"
            Basic token authentication:
            ```python
            @auth.authenticate
            async def authenticate(authorization: str) -> str:
                user_id = verify_token(authorization)
                return user_id
            ```

            Accept the full request context:
            ```python
            @auth.authenticate
            async def authenticate(
                method: str,
                path: str,
                headers: dict[str, bytes]
            ) -> str:
                user = await verify_request(method, path, headers)
                return user
            ```

            Return user name and permissions:
            ```python
            @auth.authenticate
            async def authenticate(
                method: str,
                path: str,
                headers: dict[str, bytes]
            ) -> Auth.types.MinimalUserDict:
                permissions, user = await verify_request(method, path, headers)
                # Permissions could be things like ["runs:read", "runs:write", "threads:read", "threads:write"]
                return {
                    "identity": user["id"],
                    "permissions": permissions,
                    "display_name": user["name"],
                }
            ```
        """
        if self._authenticate_handler is not None:
            raise ValueError(
                "Authentication handler already set as {self._authenticate_handler}."
            )
        self._authenticate_handler = fn
        return fn


## Helper types & utilities

V = typing.TypeVar("V", contravariant=True)


class _ActionHandler(typing.Protocol[V]):
    async def __call__(
        self, *, ctx: types.AuthContext, value: V
    ) -> types.HandlerResult: ...


T = typing.TypeVar("T", covariant=True)


class _ResourceActionOn(typing.Generic[T]):
    def __init__(
        self,
        auth: Auth,
        resource: typing.Literal["threads", "crons", "assistants"],
        action: typing.Literal[
            "create", "read", "update", "delete", "search", "create_run"
        ],
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


VCreate = typing.TypeVar("VCreate", covariant=True)
VUpdate = typing.TypeVar("VUpdate", covariant=True)
VRead = typing.TypeVar("VRead", covariant=True)
VDelete = typing.TypeVar("VDelete", covariant=True)
VSearch = typing.TypeVar("VSearch", covariant=True)


class _ResourceOn(typing.Generic[VCreate, VRead, VUpdate, VDelete, VSearch]):
    """
    Generic base class for resource-specific handlers.
    """

    value: type[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]]

    Create: type[VCreate]
    Read: type[VRead]
    Update: type[VUpdate]
    Delete: type[VDelete]
    Search: type[VSearch]

    def __init__(
        self,
        auth: Auth,
        resource: typing.Literal["threads", "crons", "assistants"],
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

    @typing.overload
    def __call__(
        self,
        fn: typing.Union[
            _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
            _ActionHandler[dict[str, typing.Any]],
        ],
    ) -> _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]]: ...

    @typing.overload
    def __call__(
        self,
        *,
        resources: typing.Union[str, Sequence[str]],
        actions: typing.Optional[typing.Union[str, Sequence[str]]] = None,
    ) -> Callable[
        [_ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]]],
        _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
    ]: ...

    def __call__(
        self,
        fn: typing.Union[
            _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
            _ActionHandler[dict[str, typing.Any]],
            None,
        ] = None,
        *,
        resources: typing.Union[str, Sequence[str], None] = None,
        actions: typing.Optional[typing.Union[str, Sequence[str]]] = None,
    ) -> typing.Union[
        _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
        Callable[
            [_ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]]],
            _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
        ],
    ]:
        if fn is not None:
            _validate_handler(fn)
            return typing.cast(
                _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
                _register_handler(self.auth, self.resource, "*", fn),
            )

        def decorator(
            handler: _ActionHandler[
                typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]
            ],
        ) -> _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]]:
            _validate_handler(handler)
            return typing.cast(
                _ActionHandler[typing.Union[VCreate, VUpdate, VRead, VDelete, VSearch]],
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
    value = typing.Union[
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
    value = typing.Union[
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
        resource: typing.Literal["threads", "crons", "assistants"],
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
        typing.Union[
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


class _StoreOn:
    def __init__(self, auth: Auth) -> None:
        self._auth = auth

    @typing.overload
    def __call__(
        self,
        *,
        actions: typing.Optional[
            typing.Union[
                typing.Literal["put", "get", "search", "list_namespaces", "delete"],
                Sequence[
                    typing.Literal["put", "get", "search", "list_namespaces", "delete"]
                ],
            ]
        ] = None,
    ) -> Callable[[AHO], AHO]: ...

    @typing.overload
    def __call__(self, fn: AHO) -> AHO: ...

    def __call__(
        self,
        fn: typing.Optional[AHO] = None,
        *,
        actions: typing.Optional[
            typing.Union[
                typing.Literal["put", "get", "search", "list_namespaces", "delete"],
                Sequence[
                    typing.Literal["put", "get", "search", "list_namespaces", "delete"]
                ],
            ]
        ] = None,
    ) -> typing.Union[AHO, Callable[[AHO], AHO]]:
        """Register a handler for specific resources and actions.

        Can be used as a decorator or with explicit resource/action parameters:

        @auth.on.store
        async def handler(): ... # Handle all store ops

        @auth.on.store(actions=("put", "get", "search", "delete"))
        async def handler(): ... # Handle specific store ops

        @auth.on.store.put
        async def handler(): ... # Handle store.put ops
        """
        if fn is not None:
            # Used as a plain decorator
            _register_handler(self._auth, "store", None, fn)
            return fn

        # Used with parameters, return a decorator
        def decorator(
            handler: AHO,
        ) -> AHO:
            if isinstance(actions, str):
                action_list = [actions]
            else:
                action_list = list(actions) if actions is not None else ["*"]
            for action in action_list:
                _register_handler(self._auth, "store", action, handler)
            return handler

        return decorator


AHO = typing.TypeVar("AHO", bound=_ActionHandler[dict[str, typing.Any]])


class _On:
    """Entry point for authorization handlers that control access to specific resources.

    The _On class provides a flexible way to define authorization rules for different resources
    and actions in your application. It supports three main usage patterns:

    1. Global handlers that run for all resources and actions
    2. Resource-specific handlers that run for all actions on a resource
    3. Resource and action specific handlers for fine-grained control

    Each handler must be an async function that accepts two parameters:
    - ctx (AuthContext): Contains request context and authenticated user info
    - value: The data being authorized (type varies by endpoint)

    The handler should return one of:
        - None or True: Accept the request
        - False: Reject with 403 error
        - FilterType: Apply filtering rules to the response

    ???+ example "Examples"

        Global handler for all requests:
        ```python
        @auth.on
        async def log_all_requests(ctx: AuthContext, value: Any) -> None:
            print(f"Request to {ctx.path} by {ctx.user.identity}")
            return True
        ```

        Resource-specific handler:
        ```python
        @auth.on.threads
        async def check_thread_access(ctx: AuthContext, value: Any) -> bool:
            # Allow access only to threads created by the user
            return value.get("created_by") == ctx.user.identity
        ```

        Resource and action specific handler:
        ```python
        @auth.on.threads.delete
        async def prevent_thread_deletion(ctx: AuthContext, value: Any) -> bool:
            # Only admins can delete threads
            return "admin" in ctx.user.permissions
        ```

        Multiple resources or actions:
        ```python
        @auth.on(resources=["threads", "runs"], actions=["create", "update"])
        async def rate_limit_writes(ctx: AuthContext, value: Any) -> bool:
            # Implement rate limiting for write operations
            return await check_rate_limit(ctx.user.identity)
        ```
    """

    __slots__ = (
        "_auth",
        "assistants",
        "threads",
        "runs",
        "crons",
        "store",
        "value",
    )

    def __init__(self, auth: Auth) -> None:
        self._auth = auth
        self.assistants = _AssistantsOn(auth, "assistants")
        self.threads = _ThreadsOn(auth, "threads")
        self.crons = _CronsOn(auth, "crons")
        self.store = _StoreOn(auth)
        self.value = dict[str, typing.Any]

    @typing.overload
    def __call__(
        self,
        *,
        resources: typing.Union[str, Sequence[str]],
        actions: typing.Optional[typing.Union[str, Sequence[str]]] = None,
    ) -> Callable[[AHO], AHO]: ...

    @typing.overload
    def __call__(self, fn: AHO) -> AHO: ...

    def __call__(
        self,
        fn: typing.Optional[AHO] = None,
        *,
        resources: typing.Union[str, Sequence[str], None] = None,
        actions: typing.Optional[typing.Union[str, Sequence[str]]] = None,
    ) -> typing.Union[AHO, Callable[[AHO], AHO]]:
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
        def decorator(
            handler: AHO,
        ) -> AHO:
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
    auth: Auth,
    resource: typing.Optional[str],
    action: typing.Optional[str],
    fn: types.Handler,
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


def _validate_handler(fn: Callable[..., typing.Any]) -> None:
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


__all__ = ["Auth", "types", "exceptions"]
