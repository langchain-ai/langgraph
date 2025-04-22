import sys
import typing
from dataclasses import is_dataclass
from typing import Any, Optional, Union

import typing_extensions
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1


def create_model(
    model_name: str,
    *,
    field_definitions: Optional[dict[str, Any]] = None,
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
