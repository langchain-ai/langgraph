import functools
import warnings
from typing import Any, Callable, Type, TypeVar, Union, cast


class LangGraphDeprecationWarning(DeprecationWarning):
    pass


F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=Type[Any])


def deprecated(
    since: str, alternative: str, *, removal: str = "", example: str = ""
) -> Callable[[F], F]:
    def decorator(obj: Union[F, C]) -> Union[F, C]:
        removal_str = removal if removal else "a future version"
        message = (
            f"{obj.__name__} is deprecated as of version {since} and will be"
            f" removed in {removal_str}. Use {alternative} instead.{example}"
        )
        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def new_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
                warnings.warn(message, LangGraphDeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[misc]

            docstring = (
                f"**Deprecated**: This class is deprecated as of version {since}. "
                f"Use `{alternative}` instead."
            )
            if obj.__doc__:
                docstring = docstring + f"\n\n{obj.__doc__}"
            obj.__doc__ = docstring

            return cast(C, obj)
        elif callable(obj):

            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(message, LangGraphDeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            docstring = (
                f"**Deprecated**: This function is deprecated as of version {since}. "
                f"Use `{alternative}` instead."
            )
            if obj.__doc__:
                docstring = docstring + f"\n\n{obj.__doc__}"
            wrapper.__doc__ = docstring

            return cast(F, wrapper)
        else:
            raise TypeError(
                f"Can only add deprecation decorator to classes or callables, got '{type(obj)}' instead."
            )

    return decorator


def deprecated_parameter(
    arg_name: str, since: str, alternative: str, *, removal: str
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
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
