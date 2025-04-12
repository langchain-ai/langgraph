import sys
import typing
from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Sequence, Union

import typing_extensions
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

# NOTE: this is redefined here separately from langgraph.constants
# to avoid a circular import
MISSING = object()


def create_model(
    model_name: str,
    *,
    field_definitions: Optional[Dict[str, Any]] = None,
    root: Optional[Any] = None,
) -> Union[BaseModel, BaseModelV1]:
    """Create a pydantic model with the given field definitions.

    Args:
        model_name: The name of the model.
        field_definitions: The field definitions for the model.
        root: Type for a root model (RootModel)
    """
    try:
        # for langchain-core >= 0.3.0
        from langchain_core.utils.pydantic import create_model_v2

        return create_model_v2(
            model_name,
            field_definitions=field_definitions,
            root=root,
        )
    except ImportError:
        # for langchain-core < 0.3.0
        from langchain_core.runnables.utils import create_model

        v1_kwargs = {}
        if root is not None:
            v1_kwargs["__root__"] = root

        return create_model(model_name, **v1_kwargs, **(field_definitions or {}))


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


def is_supported_by_pydantic(type_: Any) -> bool:
    """Check if a given "complex" type is supported by pydantic.

    This will return False for primitive types like int, str, etc.

    The check is meant for container types like dataclasses, TypedDicts, etc.
    """
    if is_dataclass(type_):
        return True

    # Pydantic does not support mixing .v1 and root namespaces, so
    # we only check for BaseModel (not pydantic.v1.BaseModel).
    if isinstance(type_, type) and issubclass(type_, BaseModel):
        return True

    if hasattr(type_, "__orig_bases__"):
        for base in type_.__orig_bases__:
            if base is typing_extensions.TypedDict:
                return True
            elif base is typing.TypedDict:  # noqa: TID251
                # ignoring TID251 since it's OK to use typing.TypedDict in this case.
                # Pydantic supports typing.TypedDict from Python 3.12
                # For older versions, only typing_extensions.TypedDict is supported.
                if sys.version_info >= (3, 12):
                    return True
    return False
