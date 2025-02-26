from pydantic import BaseModel as BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing import Any

def create_model(model_name: str, *, field_definitions: dict[str, Any] | None = None, root: Any | None = None) -> BaseModel | BaseModelV1: ...
