import dataclasses
from typing import Any, Optional, Type, Union

from typing_extensions import Annotated, NotRequired, ReadOnly, Required, get_origin


def _is_optional_type(type_: Any) -> bool:
    """Check if a type is Optional."""

    if hasattr(type_, "__origin__") and hasattr(type_, "__args__"):
        origin = get_origin(type_)
        if origin is Optional:
            return True
        if origin is Union:
            return any(
                arg is type(None) or _is_optional_type(arg) for arg in type_.__args__
            )
        if origin is Annotated:
            return _is_optional_type(type_.__args__[0])
        return origin is None
    if hasattr(type_, "__bound__") and type_.__bound__ is not None:
        return _is_optional_type(type_.__bound__)
    return type_ is None


def _is_required_type(type_: Any) -> Optional[bool]:
    """Check if an annotation is marked as Required/NotRequired.

    Returns:
        - True if required
        - False if not required
        - None if not annotated with either
    """
    origin = get_origin(type_)
    if origin is Required:
        return True
    if origin is NotRequired:
        return False
    if origin is Annotated or getattr(origin, "__args__", None):
        # See https://typing.readthedocs.io/en/latest/spec/typeddict.html#interaction-with-annotated
        return _is_required_type(type_.__args__[0])
    return None


def _is_readonly_type(type_: Any) -> bool:
    """Check if an annotation is marked as ReadOnly.

    Returns:
        - True if is read only
        - False if not read only
    """

    # See: https://typing.readthedocs.io/en/latest/spec/typeddict.html#typing-readonly-type-qualifier
    origin = get_origin(type_)
    if origin is Annotated:
        return _is_readonly_type(type_.__args__[0])
    if origin is ReadOnly:
        return True
    return False


_DEFAULT_KEYS = frozenset()


def get_field_default(name: str, type_: Any, schema: Type[Any]) -> Any:
    """Determine the default value for a field in a state schema.

    This is based on:
        If TypedDict:
            - Required/NotRequired
            - total=False -> everything optional
        - Type annotation (Optional/Union[None])
    """
    optional_keys = getattr(schema, "__optional_keys__", _DEFAULT_KEYS)
    irq = _is_required_type(type_)
    if name in optional_keys:
        # Either total=False or explicit NotRequired.
        # No type annotation trumps this.
        if irq:
            # Unless it's earlier versions of python & explicit Required
            return ...
        return None
    if irq is not None:
        if irq:
            # Handle Required[<type>]
            # (we already handled NotRequired and total=False)
            return ...
        # Handle NotRequired[<type>] for earlier versions of python
        return None
    if dataclasses.is_dataclass(schema):
        field_info = next(
            (f for f in dataclasses.fields(schema) if f.name == name), None
        )
        if field_info:
            if (
                field_info.default is not dataclasses.MISSING
                and field_info.default is not ...
            ):
                return field_info.default
            elif field_info.default_factory is not dataclasses.MISSING:
                return field_info.default_factory()
    # Note, we ignore ReadOnly attributes,
    # as they don't make much sense. (we don't care if you mutate the state in your node)
    # and mutating state in your node has no effect on our graph state.
    # Base case is the annotation
    if _is_optional_type(type_):
        return None
    return ...
