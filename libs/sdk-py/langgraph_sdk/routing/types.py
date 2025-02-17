"""Type-hints for the langgraph router.

Copied from Starlette. When implementing, use Starlette types."""

import enum
import typing

from typing_extensions import ParamSpec


class Match(enum.Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


AppType = typing.TypeVar("AppType")
Scope = typing.MutableMapping[str, typing.Any]
Message = typing.MutableMapping[str, typing.Any]
Receive = typing.Callable[[], typing.Awaitable[Message]]
Send = typing.Callable[[Message], typing.Awaitable[None]]


@typing.runtime_checkable
class BaseRoute(typing.Protocol):
    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        """Determine if the route matches the given scope."""
        ...

    def url_path_for(self, name: str, /, **path_params: typing.Any) -> str:
        """Return the URL path for the given name and path parameters."""
        ...

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle the event."""
        ...

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle the event."""
        ...


StatelessLifespan = typing.Callable[[AppType], typing.AsyncContextManager[None]]
StatefulLifespan = typing.Callable[
    [AppType], typing.AsyncContextManager[typing.Mapping[str, typing.Any]]
]
Lifespan = typing.Union[StatelessLifespan[AppType], StatefulLifespan[AppType]]
ASGIApp = typing.Callable[[Scope, Receive, Send], typing.Awaitable[None]]


P = ParamSpec("P")


class _MiddlewareFactory(typing.Protocol[P]):
    def __call__(
        self, app: ASGIApp, /, *args: P.args, **kwargs: P.kwargs
    ) -> ASGIApp: ...  # pragma: no cover


# Copied from Starlette. Basically a named tuple
class Middleware:
    def __init__(
        self,
        cls: _MiddlewareFactory[P],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> typing.Iterator[typing.Any]:
        as_tuple = (self.cls, self.args, self.kwargs)
        return iter(as_tuple)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        args_strings = [f"{value!r}" for value in self.args]
        option_strings = [f"{key}={value!r}" for key, value in self.kwargs.items()]
        name = getattr(self.cls, "__name__", "")
        args_repr = ", ".join([name] + args_strings + option_strings)
        return f"{class_name}({args_repr})"
