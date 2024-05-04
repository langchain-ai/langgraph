import functools
import warnings
from typing import Callable, TypeVar


class LangGraphDeprecationWarning(DeprecationWarning):
    pass


F = TypeVar("F", bound=Callable)


def deprecated(version: str, alternative: str, *, example: str = ""):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = (
                f"{func.__name__} is deprecated as of version {version} and will be"
                f" removed in a future version. Use {alternative} instead.{example}"
            )
            warnings.warn(message, LangGraphDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        docstring = (
            f"**Deprecated**: This function is deprecated as of version {version}. "
            f"Use `{alternative}` instead."
        )
        if func.__doc__:
            docstring = docstring + f"\n\n{func.__doc__}"
        wrapper.__doc__ = docstring

        return wrapper

    return decorator
