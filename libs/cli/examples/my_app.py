from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

my_context_var: ContextVar[str] = ContextVar("my_context_var", default="")
LIFESPAN_VAL = ""
other_context_var = ContextVar("other_context_var", default="")


@asynccontextmanager
async def my_lifespan(app):
    global LIFESPAN_VAL
    LIFESPAN_VAL = "foobar-lifespan"
    yield
    assert LIFESPAN_VAL == "foobar-lifespan"
    LIFESPAN_VAL = ""


class MyContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Any, call_next: Any) -> Any:
        token = my_context_var.set("Foobar")
        try:
            response = await call_next(request)
            return response
        finally:
            my_context_var.reset(token)


async def custom_my_route(request):
    """A great route."""
    assert my_context_var.get() == "Foobar"
    assert LIFESPAN_VAL == "foobar-lifespan"
    return JSONResponse({"foo": "bar"})


async def runs_afakeroute(request):
    """Another great route."""
    assert my_context_var.get() == "Foobar"
    assert LIFESPAN_VAL == "foobar-lifespan"
    return JSONResponse({"foo": "afakeroute"})


async def other_middleware(request: Any, call_next: Any) -> Any:
    other_context_var.set("foobar")
    response = await call_next(request)
    other_context_var.reset()
    return response


app = Starlette(
    middleware=[(MyContextMiddleware, {}, {})],
    routes=[
        Route("/custom/my-route", custom_my_route),
        Route("/runs/afakeroute", runs_afakeroute),
    ],
    lifespan=my_lifespan,
)
