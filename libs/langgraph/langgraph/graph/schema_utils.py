import logging
import weakref
from typing import (
    Any,
    Callable,
    Optional,
    Type,
)

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

logger = logging.getLogger(__name__)


_cache: weakref.WeakKeyDictionary[Type[Any], "SchemaCoercionMapper"] = (
    weakref.WeakKeyDictionary()
)


class SchemaCoercionMapper:
    __slots__ = ("_inited", "schema", "_fields", "_construct", "_field_coercers")

    def __new__(
        cls,
        schema: Type[Any],
        **kwargs: Any,
    ) -> "SchemaCoercionMapper":
        if schema in _cache:
            return _cache[schema]
        inst = super().__new__(cls)
        _cache[schema] = inst
        return inst

    def __init__(
        self,
        schema: Type[Any],
        **kwargs: Any,
    ):
        if hasattr(self, "_inited"):
            return
        self._inited = True
        if issubclass(schema, BaseModelV1):
            self._construct: Callable[..., Any] = schema.parse_obj

        elif issubclass(schema, BaseModel):
            self._construct = schema.model_validate

        else:
            raise TypeError("Schema is neither valid Pydantic v1 nor v2 model.")

    def __call__(self, input_data: Any, depth: Optional[int] = None) -> Any:
        if not isinstance(input_data, dict):
            return input_data
        return self._construct(input_data)
