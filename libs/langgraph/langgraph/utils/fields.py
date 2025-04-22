import dataclasses
from collections.abc import Generator, Sequence
from typing import Annotated, Any, Optional, Union, get_type_hints

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import NotRequired, ReadOnly, Required, get_origin

# NOTE: this is redefined here separately from langgraph.constants
# to avoid a circular import
MISSING = object()


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


_DEFAULT_KEYS: frozenset[str] = frozenset()


def get_field_default(name: str, type_: Any, schema: type[Any]) -> Any:
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


def get_enhanced_type_hints(
    type: type[Any],
) -> Generator[tuple[str, Any, Any, Optional[str]], None, None]:
    """Attempt to extract default values and descriptions from provided type, used for config schema."""
    for name, typ in get_type_hints(type).items():
        default = None
        description = None

        # Pydantic models
        try:
            if hasattr(type, "__fields__") and name in type.__fields__:
                field = type.__fields__[name]

                if hasattr(field, "description") and field.description is not None:
                    description = field.description

                if hasattr(field, "default") and field.default is not None:
                    default = field.default
                    if (
                        hasattr(default, "__class__")
                        and getattr(default.__class__, "__name__", "")
                        == "PydanticUndefinedType"
                    ):
                        default = None

        except (AttributeError, KeyError, TypeError):
            pass

        # TypedDict, dataclass
        try:
            if hasattr(type, "__dict__"):
                type_dict = getattr(type, "__dict__")

                if name in type_dict:
                    default = type_dict[name]
        except (AttributeError, KeyError, TypeError):
            pass

        yield name, typ, default, description


def get_update_as_tuples(input: Any, keys: Sequence[str]) -> list[tuple[str, Any]]:
    """Get Pydantic state update as a list of (key, value) tuples."""
    # Pydantic v1
    if isinstance(input, BaseModelV1):
        keep: Optional[set[str]] = input.__fields_set__
        defaults = {k: v.default for k, v in input.__fields__.items()}
    # Pydantic v2
    elif isinstance(input, BaseModel):
        keep = input.model_fields_set
        defaults = {k: v.default for k, v in input.model_fields.items()}
    else:
        keep = None
        defaults = {}

    # NOTE: This behavior for Pydantic is somewhat inelegant,
    # but we keep around for backwards compatibility
    # if input is a Pydantic model, only update values
    # that are different from the default values or in the keep set
    return [
        (k, value)
        for k in keys
        if (value := getattr(input, k, MISSING)) is not MISSING
        and (
            value is not None
            or defaults.get(k, MISSING) is not None
            or (keep is not None and k in keep)
        )
    ]
