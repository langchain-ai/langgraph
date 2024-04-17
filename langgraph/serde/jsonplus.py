import importlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from langchain_core.load.load import Reviver
from langchain_core.load.serializable import Serializable
from langchain_core.pydantic_v1 import BaseModel as LcBaseModel
from pydantic import BaseModel

from langgraph.serde.base import SerializerProtocol

LC_REVIVER = Reviver()


class JsonPlusSerializer(SerializerProtocol):
    def _encode_constructor_args(
        self,
        constructor: type[Any],
        *,
        method: Optional[str] = None,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        return {
            "lc": 2,
            "type": "constructor",
            "id": [*constructor.__module__.split("."), constructor.__name__],
            "method": method,
            "args": args if args is not None else [],
            "kwargs": kwargs if kwargs is not None else {},
        }

    def _default(self, obj):
        if isinstance(obj, Serializable):
            return obj.to_json()
        elif isinstance(obj, (BaseModel, LcBaseModel)):
            return self._encode_constructor_args(obj.__class__, kwargs=obj.dict())
        elif isinstance(obj, UUID):
            return self._encode_constructor_args(UUID, args=[obj.hex])
        elif isinstance(obj, (set, frozenset)):
            return self._encode_constructor_args(type(obj), args=[list(obj)])
        elif isinstance(obj, datetime):
            return self._encode_constructor_args(
                datetime, method="fromisoformat", args=[obj.isoformat(), obj.tzinfo]
            )
        else:
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

    def _reviver(self, value: dict[str, Any]) -> Any:
        if (
            value.get("lc", None) == 2
            and value.get("type", None) == "constructor"
            and value.get("id", None) is not None
        ):
            # Get module and class name
            [*module, name] = value["id"]
            # Import module
            mod = importlib.import_module(".".join(module))
            # Import class
            cls = getattr(mod, name)
            # Instantiate class
            if value["method"] is not None:
                method = getattr(cls, value["method"])
                return method(*value["args"], **value["kwargs"])
            else:
                return cls(*value["args"], **value["kwargs"])

        return LC_REVIVER(value)

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, default=self._default, sort_keys=True)

    def loads(self, data: bytes) -> Any:
        return json.loads(data)
