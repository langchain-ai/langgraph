from typing import Any, Dict, Optional, Union

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1


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
        from langchain_core.runnables.pydantic import create_model_v2

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
