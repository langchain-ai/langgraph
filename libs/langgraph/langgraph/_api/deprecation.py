import functools
import warnings
from typing import Any, Callable, TypeVar, cast


class LangGraphDeprecationWarning(DeprecationWarning):
    pass


F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    since: str, alternative: str, *, removal: str = "", example: str = ""
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            removal_str = removal if removal else "a future version"
            message = (
                f"{func.__name__} is deprecated as of version {since} and will be"
                f" removed in {removal_str}. Use {alternative} instead.{example}"
            )
            warnings.warn(message, LangGraphDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        docstring = (
            f"**Deprecated**: This function is deprecated as of version {since}. "
            f"Use `{alternative}` instead."
        )
        if func.__doc__:
            docstring = docstring + f"\n\n{func.__doc__}"
        wrapper.__doc__ = docstring

        return cast(F, wrapper)

    return decorator


def deprecated_parameter(
    arg_name: str, since: str, alternative: str, *, removal: str
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name in kwargs:
                warnings.warn(
                    f"Parameter '{arg_name}' in function '{func.__name__}' is "
                    f"deprecated as of version {since} and will be removed in version {removal}. "
                    f"Use '{alternative}' parameter instead.",
                    category=LangGraphDeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
