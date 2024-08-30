import dataclasses
import decimal
import importlib
import json
import pathlib
import re
from collections import deque
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from inspect import isclass
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from typing import Any, Optional
from uuid import UUID

from langchain_core.load.load import Reviver
from langchain_core.load.serializable import Serializable
from zoneinfo import ZoneInfo

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import SendProtocol

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
        elif hasattr(obj, "model_dump") and callable(obj.model_dump):
            return self._encode_constructor_args(
                obj.__class__, method=[None, "model_construct"], kwargs=obj.model_dump()
            )
        elif hasattr(obj, "dict") and callable(obj.dict):
            return self._encode_constructor_args(
                obj.__class__, method=[None, "construct"], kwargs=obj.dict()
            )
        elif isinstance(obj, pathlib.Path):
            return self._encode_constructor_args(pathlib.Path, args=obj.parts)
        elif isinstance(obj, re.Pattern):
            return self._encode_constructor_args(
                re.compile, args=[obj.pattern, obj.flags]
            )
        elif isinstance(obj, UUID):
            return self._encode_constructor_args(UUID, args=[obj.hex])
        elif isinstance(obj, decimal.Decimal):
            return self._encode_constructor_args(decimal.Decimal, args=[str(obj)])
        elif isinstance(obj, (set, frozenset, deque)):
            return self._encode_constructor_args(type(obj), args=[list(obj)])
        elif isinstance(obj, (IPv4Address, IPv4Interface, IPv4Network)):
            return self._encode_constructor_args(obj.__class__, args=[str(obj)])
        elif isinstance(obj, (IPv6Address, IPv6Interface, IPv6Network)):
            return self._encode_constructor_args(obj.__class__, args=[str(obj)])

        elif isinstance(obj, datetime):
            return self._encode_constructor_args(
                datetime, method="fromisoformat", args=[obj.isoformat()]
            )
        elif isinstance(obj, timezone):
            return self._encode_constructor_args(timezone, args=obj.__getinitargs__())
        elif isinstance(obj, ZoneInfo):
            return self._encode_constructor_args(ZoneInfo, args=[obj.key])
        elif isinstance(obj, timedelta):
            return self._encode_constructor_args(
                timedelta, args=[obj.days, obj.seconds, obj.microseconds]
            )
        elif isinstance(obj, date):
            return self._encode_constructor_args(
                date, args=[obj.year, obj.month, obj.day]
            )
        elif isinstance(obj, time):
            return self._encode_constructor_args(
                time,
                args=[obj.hour, obj.minute, obj.second, obj.microsecond, obj.tzinfo],
                kwargs={"fold": obj.fold},
            )
        elif dataclasses.is_dataclass(obj):
            return self._encode_constructor_args(
                obj.__class__,
                kwargs={
                    field.name: getattr(obj, field.name)
                    for field in dataclasses.fields(obj)
                },
            )
        elif isinstance(obj, Enum):
            return self._encode_constructor_args(obj.__class__, args=[obj.value])
        elif isinstance(obj, SendProtocol):
            return self._encode_constructor_args(
                obj.__class__, kwargs={"node": obj.node, "arg": obj.arg}
            )
        elif isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=[obj.hex()]
            )
        elif isinstance(obj, BaseException):
            return repr(obj)
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
            try:
                # Get module and class name
                [*module, name] = value["id"]
                # Import module
                mod = importlib.import_module(".".join(module))
                # Import class
                cls = getattr(mod, name)
                # Instantiate class
                if isinstance(value["method"], str):
                    methods = [getattr(cls, value["method"])]
                elif isinstance(value["method"], list):
                    methods = [
                        cls if method is None else getattr(cls, method)
                        for method in value["method"]
                    ]
                else:
                    methods = [cls]
                for method in methods:
                    try:
                        if isclass(method) and issubclass(method, BaseException):
                            return None
                        if value["args"] and value["kwargs"]:
                            return method(*value["args"], **value["kwargs"])
                        elif value["args"]:
                            return method(*value["args"])
                        elif value["kwargs"]:
                            return method(**value["kwargs"])
                        else:
                            return method()
                    except Exception:
                        continue
            except Exception:
                return None

        return LC_REVIVER(value)

    def dumps(self, obj: Any) -> bytes:
        return json.dumps(obj, default=self._default, ensure_ascii=False).encode(
            "utf-8", "ignore"
        )

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj
        else:
            return "json", self.dumps(obj)

    def loads(self, data: bytes) -> Any:
        return json.loads(data, object_hook=self._reviver)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "bytes":
            return data_
        elif type_ == "bytearray":
            return bytearray(data_)
        elif type_ == "json":
            return self.loads(data_)
        else:
            raise NotImplementedError(f"Unknown serialization type: {type_}")
