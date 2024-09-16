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
from typing import Any, Optional, Sequence
from uuid import UUID

import msgpack
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
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        out = {
            "lc": 2,
            "type": "constructor",
            "id": (*constructor.__module__.split("."), constructor.__name__),
        }
        if method is not None:
            out["method"] = method
        if args is not None:
            out["args"] = args
        if kwargs is not None:
            out["kwargs"] = kwargs
        return out

    def _default(self, obj):
        if isinstance(obj, Serializable):
            return obj.to_json()
        elif hasattr(obj, "model_dump") and callable(obj.model_dump):
            return self._encode_constructor_args(
                obj.__class__, method=(None, "model_construct"), kwargs=obj.model_dump()
            )
        elif hasattr(obj, "dict") and callable(obj.dict):
            return self._encode_constructor_args(
                obj.__class__, method=(None, "construct"), kwargs=obj.dict()
            )
        elif hasattr(obj, "_asdict") and callable(obj._asdict):
            return self._encode_constructor_args(obj.__class__, kwargs=obj._asdict())
        elif isinstance(obj, pathlib.Path):
            return self._encode_constructor_args(pathlib.Path, args=obj.parts)
        elif isinstance(obj, re.Pattern):
            return self._encode_constructor_args(
                re.compile, args=(obj.pattern, obj.flags)
            )
        elif isinstance(obj, UUID):
            return self._encode_constructor_args(UUID, args=(obj.hex,))
        elif isinstance(obj, decimal.Decimal):
            return self._encode_constructor_args(decimal.Decimal, args=(str(obj),))
        elif isinstance(obj, (set, frozenset, deque)):
            return self._encode_constructor_args(type(obj), args=(tuple(obj),))
        elif isinstance(obj, (IPv4Address, IPv4Interface, IPv4Network)):
            return self._encode_constructor_args(obj.__class__, args=(str(obj),))
        elif isinstance(obj, (IPv6Address, IPv6Interface, IPv6Network)):
            return self._encode_constructor_args(obj.__class__, args=(str(obj),))

        elif isinstance(obj, datetime):
            return self._encode_constructor_args(
                datetime, method="fromisoformat", args=(obj.isoformat(),)
            )
        elif isinstance(obj, timezone):
            return self._encode_constructor_args(timezone, args=obj.__getinitargs__())
        elif isinstance(obj, ZoneInfo):
            return self._encode_constructor_args(ZoneInfo, args=(obj.key,))
        elif isinstance(obj, timedelta):
            return self._encode_constructor_args(
                timedelta, args=(obj.days, obj.seconds, obj.microseconds)
            )
        elif isinstance(obj, date):
            return self._encode_constructor_args(
                date, args=(obj.year, obj.month, obj.day)
            )
        elif isinstance(obj, time):
            return self._encode_constructor_args(
                time,
                args=(obj.hour, obj.minute, obj.second, obj.microsecond, obj.tzinfo),
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
            return self._encode_constructor_args(obj.__class__, args=(obj.value,))
        elif isinstance(obj, SendProtocol):
            return self._encode_constructor_args(
                obj.__class__, kwargs={"node": obj.node, "arg": obj.arg}
            )
        elif isinstance(obj, (bytes, bytearray)):
            return self._encode_constructor_args(
                obj.__class__, method="fromhex", args=(obj.hex(),)
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
                method = value.get("method")
                if isinstance(method, str):
                    methods = [getattr(cls, method)]
                elif isinstance(method, list):
                    methods = [
                        cls if method is None else getattr(cls, method)
                        for method in method
                    ]
                else:
                    methods = [cls]
                args = value.get("args")
                kwargs = value.get("kwargs")
                for method in methods:
                    try:
                        if isclass(method) and issubclass(method, BaseException):
                            return None
                        if args and kwargs:
                            return method(*args, **kwargs)
                        elif args:
                            return method(*args)
                        elif kwargs:
                            return method(**kwargs)
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
            try:
                return "msgpack", _msgpack_enc(obj)
            except UnicodeEncodeError:
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
        elif type_ == "msgpack":
            return msgpack.unpackb(data_, ext_hook=_msgpack_ext_hook)
        else:
            raise NotImplementedError(f"Unknown serialization type: {type_}")


# --- msgpack ---

EXT_CONSTRUCTOR_SINGLE_ARG = 0
EXT_CONSTRUCTOR_POS_ARGS = 1
EXT_CONSTRUCTOR_KW_ARGS = 2
EXT_METHOD_SINGLE_ARG = 3
EXT_PYDANTIC_V1 = 4
EXT_PYDANTIC_V2 = 5


def _msgpack_default(obj):
    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # pydantic v2
        return msgpack.ExtType(
            EXT_PYDANTIC_V2,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.model_dump(),
                    "model_validate_json",
                ),
            ),
        )
    elif hasattr(obj, "dict") and callable(obj.dict):  # pydantic v1
        return msgpack.ExtType(
            EXT_PYDANTIC_V1,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.dict(),
                ),
            ),
        )
    elif hasattr(obj, "_asdict") and callable(obj._asdict):  # namedtuple
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_KW_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj._asdict(),
                ),
            ),
        )
    elif isinstance(obj, pathlib.Path):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.parts),
            ),
        )
    elif isinstance(obj, re.Pattern):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                ("re", "compile", (obj.pattern, obj.flags)),
            ),
        )
    elif isinstance(obj, UUID):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.hex),
            ),
        )
    elif isinstance(obj, decimal.Decimal):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, (set, frozenset, deque)):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, tuple(obj)),
            ),
        )
    elif isinstance(obj, (IPv4Address, IPv4Interface, IPv4Network)):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, (IPv6Address, IPv6Interface, IPv6Network)):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, datetime):
        return msgpack.ExtType(
            EXT_METHOD_SINGLE_ARG,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.isoformat(),
                    "fromisoformat",
                ),
            ),
        )
    elif isinstance(obj, timedelta):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    (obj.days, obj.seconds, obj.microseconds),
                ),
            ),
        )
    elif isinstance(obj, date):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    (obj.year, obj.month, obj.day),
                ),
            ),
        )
    elif isinstance(obj, time):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_KW_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    {
                        "hour": obj.hour,
                        "minute": obj.minute,
                        "second": obj.second,
                        "microsecond": obj.microsecond,
                        "tzinfo": obj.tzinfo,
                        "fold": obj.fold,
                    },
                ),
            ),
        )
    elif isinstance(obj, timezone):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.__getinitargs__(),
                ),
            ),
        )
    elif isinstance(obj, ZoneInfo):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.key),
            ),
        )
    elif isinstance(obj, Enum):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.value),
            ),
        )
    elif isinstance(obj, SendProtocol):
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, (obj.node, obj.arg)),
            ),
        )
    elif dataclasses.is_dataclass(obj):
        # doesn't use dataclasses.asdict to avoid deepcopy and recursion
        return msgpack.ExtType(
            EXT_CONSTRUCTOR_KW_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    {
                        field.name: getattr(obj, field.name)
                        for field in dataclasses.fields(obj)
                    },
                ),
            ),
        )
    elif isinstance(obj, BaseException):
        return repr(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")


def _msgpack_ext_hook(code: int, data: bytes):
    if code == EXT_CONSTRUCTOR_SINGLE_ARG:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, arg
            return getattr(importlib.import_module(tup[0]), tup[1])(tup[2])
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_POS_ARGS:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, args
            return getattr(importlib.import_module(tup[0]), tup[1])(*tup[2])
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_KW_ARGS:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, args
            return getattr(importlib.import_module(tup[0]), tup[1])(**tup[2])
        except Exception:
            return
    elif code == EXT_METHOD_SINGLE_ARG:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, arg, method
            return getattr(getattr(importlib.import_module(tup[0]), tup[1]), tup[3])(
                tup[2]
            )
        except Exception:
            return
    elif code == EXT_PYDANTIC_V1:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, kwargs
            cls = getattr(importlib.import_module(tup[0]), tup[1])
            try:
                return cls(**tup[2])
            except Exception:
                return cls.construct(**tup[2])
        except Exception:
            return
    elif code == EXT_PYDANTIC_V2:
        try:
            tup = msgpack.unpackb(data, ext_hook=_msgpack_ext_hook)
            # module, name, kwargs, method
            cls = getattr(importlib.import_module(tup[0]), tup[1])
            try:
                return cls(**tup[2])
            except Exception:
                return cls.model_construct(**tup[2])
        except Exception:
            return


ENC_POOL = deque(maxlen=32)


def _msgpack_enc(data: Any) -> bytes:
    try:
        enc = ENC_POOL.popleft()
    except IndexError:
        enc = msgpack.Packer(default=_msgpack_default)
    try:
        return enc.pack(data)
    finally:
        ENC_POOL.append(enc)
