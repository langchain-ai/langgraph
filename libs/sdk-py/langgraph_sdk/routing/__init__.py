import typing

from langgraph_sdk.routing import types
from langgraph_sdk.routing.types import Middleware


@typing.final
class Router:
    """Add routes, middleware, and manage application lifecycle.

    Define custom routes, apply middleware globally, and handle application startup/shutdown.
    Middleware runs on all routes (including default LangGraph endpoints like /runs/, /assistants/, etc).
    Custom routes take precedence over default ones, so you can override default behavior if needed.

    ???+ example "Basic Usage"
        ```python
        from contextvars import ContextVar
        from typing import Any
        from starlette.middleware import Middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from langgraph_sdk import Router, Middleware

        # Enterprise authentication middleware example
        class JWTAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Any, call_next: Any) -> Any:
                try:
                    auth_header = request.headers["Authorization"]
                    token = auth_header.split("Bearer ")[1]
                    # Verify JWT token here
                    # Set user context for downstream handlers
                    request.state.user = {"id": "user_123"}
                    response = await call_next(request)
                    return response
                except Exception:
                    return JSONResponse(
                        {"error": "Invalid or missing authentication"},
                        status_code=401
                    )

        # Database connection pool in lifespan
        async def db_lifespan(app):
            # Initialize database connection pool
            from databases import Database
            database = Database("postgresql://user:pass@localhost/dbname")
            await database.connect()
            yield
            await database.disconnect()

        # Custom login endpoint
        async def login(request):
            data = await request.json()
            # Verify credentials and generate JWT
            return JSONResponse({
                "token": "generated.jwt.token"
            })

        # Protected endpoint example
        async def protected_route(request):
            user = request.state.user
            return JSONResponse({
                "message": f"Hello {user['id']}"
            })

        router = Router(
            middleware=[Middleware(JWTAuthMiddleware)],
            lifespan=db_lifespan,
            routes=[
                Route("/auth/login", endpoint=login, methods=["POST"]),
                Route("/api/protected", endpoint=protected_route, methods=["GET"])
            ]
        )

    ???+ note "Request Processing Flow"
        1. Middleware is applied in the order specified, wrapping all routes
        2. Routes are matched in the following order:
           * Custom routes defined in the Router take precedence
           * Default LangGraph routes are used as fallback
        3. Lifespan manages application startup/shutdown:
           * Runs before any requests are processed
           * Ideal for initializing shared resources (DB pools, caches, etc.)
           * Cleanup occurs during application shutdown

        This allows you to maintain enterprise-grade features while leveraging
        LangGraph's built-in capabilities.
    """

    __slots__ = ("routes", "lifespan", "middleware")

    def __init__(
        self,
        routes: list[types.BaseRoute],
        *,
        lifespan: typing.Union[types.Lifespan[typing.Any], None] = None,
        middleware: typing.Union[
            list[
                typing.Union[
                    types.Middleware,
                    tuple[types._MiddlewareFactory, typing.Any, typing.Any],
                ]
            ],
            None,
        ] = None,
    ) -> None:
        self.routes = routes
        self.lifespan = lifespan
        self.middleware: list[types.Middleware[typing.Any]] = middleware or []


__all__ = ["Router", "Middleware"]
