from __future__ import annotations

import dataclasses
import decimal
import importlib
import json
import logging
import pathlib
import pickle
import re
import sys
from collections import deque
from collections.abc import Callable, Sequence
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
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

import ormsgpack
from langchain_core.load.load import Reviver

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import SendProtocol
from langgraph.store.base import Item

LC_REVIVER = Reviver()
EMPTY_BYTES = b""
logger = logging.getLogger(__name__)


class JsonPlusSerializer(SerializerProtocol):
    """Serializer that uses ormsgpack, with optional fallbacks.

    Security note: this serializer is intended for use within the BaseCheckpointSaver
    class and called within the Pregel loop. It should not be used on untrusted
    python objects. If an attacker can write directly to your checkpoint database,
    they may be able to trigger code execution when data is deserialized.
    """

    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        allowed_json_modules: Sequence[tuple[str, ...]] | Literal[True] | None = None,
        __unpack_ext_hook__: Callable[[int, bytes], Any] | None = None,
    ) -> None:
        self.pickle_fallback = pickle_fallback
        self._allowed_modules = (
            {mod_and_name for mod_and_name in allowed_json_modules}
            if allowed_json_modules and allowed_json_modules is not True
            else (allowed_json_modules if allowed_json_modules is True else None)
        )
        self._unpack_ext_hook = (
            __unpack_ext_hook__
            if __unpack_ext_hook__ is not None
            else _msgpack_ext_hook
        )

    def _encode_constructor_args(
        self,
        constructor: Callable | type[Any],
        *,
        method: None | str | Sequence[None | str] = None,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

    def _reviver(self, value: dict[str, Any]) -> Any:
        if self._allowed_modules and (
            value.get("lc", None) == 2
            and value.get("type", None) == "constructor"
            and value.get("id", None) is not None
        ):
            try:
                return self._revive_lc2(value)
            except InvalidModuleError as e:
                logger.warning(
                    "Object %s is not in the deserialization allowlist.\n%s",
                    value["id"],
                    e.message,
                )

        return LC_REVIVER(value)

    def _revive_lc2(self, value: dict[str, Any]) -> Any:
        self._check_allowed_modules(value)

        [*module, name] = value["id"]
        try:
            mod = importlib.import_module(".".join(module))
            cls = getattr(mod, name)
            method = value.get("method")
            if isinstance(method, str):
                methods = [getattr(cls, method)]
            elif isinstance(method, list):
                methods = [cls if m is None else getattr(cls, m) for m in method]
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

    def _check_allowed_modules(self, value: dict[str, Any]) -> None:
        needed = tuple(value["id"])
        method = value.get("method")
        if isinstance(method, list):
            method_display = ",".join(m or "<init>" for m in method)
        elif isinstance(method, str):
            method_display = method
        else:
            method_display = "<init>"

        dotted = ".".join(needed)
        if not self._allowed_modules:
            raise InvalidModuleError(
                f"Refused to deserialize JSON constructor: {dotted} (method: {method_display}). "
                "No allowed_json_modules configured.\n\n"
                "Unblock with ONE of:\n"
                f"  • JsonPlusSerializer(allowed_json_modules=[{needed!r}, ...])\n"
                "  • (DANGEROUS) JsonPlusSerializer(allowed_json_modules=True)\n\n"
                "Note: Prefix allowlists are intentionally unsupported; prefer exact symbols "
                "or plain-JSON representations revived without import-time side effects."
            )

        if self._allowed_modules is True:
            return
        if needed in self._allowed_modules:
            return

        raise InvalidModuleError(
            f"Refused to deserialize JSON constructor: {dotted} (method: {method_display}). "
            "Symbol is not in the deserialization allowlist.\n\n"
            "Add exactly this symbol to unblock:\n"
            f"  JsonPlusSerializer(allowed_json_modules=[{needed!r}, ...])\n"
            "Or, as a last resort (DANGEROUS):\n"
            "  JsonPlusSerializer(allowed_json_modules=True)"
        )

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None:
            return "null", EMPTY_BYTES
        elif isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj
        else:
            try:
                return "msgpack", _msgpack_enc(obj)
            except ormsgpack.MsgpackEncodeError as exc:
                if self.pickle_fallback:
                    return "pickle", pickle.dumps(obj)
                raise exc

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "null":
            return None
        elif type_ == "bytes":
            return data_
        elif type_ == "bytearray":
            return bytearray(data_)
        elif type_ == "json":
            return json.loads(data_, object_hook=self._reviver)
        elif type_ == "msgpack":
            return ormsgpack.unpackb(
                data_, ext_hook=self._unpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
        elif self.pickle_fallback and type_ == "pickle":
            return pickle.loads(data_)
        else:
            raise NotImplementedError(f"Unknown serialization type: {type_}")


# --- msgpack ---

EXT_CONSTRUCTOR_SINGLE_ARG = 0
EXT_CONSTRUCTOR_POS_ARGS = 1
EXT_CONSTRUCTOR_KW_ARGS = 2
EXT_METHOD_SINGLE_ARG = 3
EXT_PYDANTIC_V1 = 4
EXT_PYDANTIC_V2 = 5
EXT_NUMPY_ARRAY = 6


def _msgpack_default(obj: Any) -> str | ormsgpack.Ext:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # pydantic v2
        return ormsgpack.Ext(
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
    elif hasattr(obj, "get_secret_value") and callable(obj.get_secret_value):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.get_secret_value(),
                ),
            ),
        )
    elif hasattr(obj, "dict") and callable(obj.dict):  # pydantic v1
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.parts),
            ),
        )
    elif isinstance(obj, re.Pattern):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                ("re", "compile", (obj.pattern, obj.flags)),
            ),
        )
    elif isinstance(obj, UUID):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.hex),
            ),
        )
    elif isinstance(obj, decimal.Decimal):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, (set, frozenset, deque)):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, tuple(obj)),
            ),
        )
    elif isinstance(obj, (IPv4Address, IPv4Interface, IPv4Network)):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, (IPv6Address, IPv6Interface, IPv6Network)):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, str(obj)),
            ),
        )
    elif isinstance(obj, datetime):
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
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
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    obj.__getinitargs__(),  # type: ignore[attr-defined]
                ),
            ),
        )
    elif isinstance(obj, ZoneInfo):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.key),
            ),
        )
    elif isinstance(obj, Enum):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_SINGLE_ARG,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, obj.value),
            ),
        )
    elif isinstance(obj, SendProtocol):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc(
                (obj.__class__.__module__, obj.__class__.__name__, (obj.node, obj.arg)),
            ),
        )
    elif dataclasses.is_dataclass(obj):
        # doesn't use dataclasses.asdict to avoid deepcopy and recursion
        return ormsgpack.Ext(
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
    elif isinstance(obj, Item):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_KW_ARGS,
            _msgpack_enc(
                (
                    obj.__class__.__module__,
                    obj.__class__.__name__,
                    {k: getattr(obj, k) for k in obj.__slots__},
                ),
            ),
        )
    elif (np_mod := sys.modules.get("numpy")) is not None and isinstance(
        obj, np_mod.ndarray
    ):
        order = "F" if obj.flags.f_contiguous and not obj.flags.c_contiguous else "C"
        if obj.flags.c_contiguous:
            mv = memoryview(obj)
            try:
                meta = (obj.dtype.str, obj.shape, order, mv)
                return ormsgpack.Ext(EXT_NUMPY_ARRAY, _msgpack_enc(meta))
            finally:
                mv.release()
        else:
            buf = obj.tobytes(order="A")
            meta = (obj.dtype.str, obj.shape, order, buf)
            return ormsgpack.Ext(EXT_NUMPY_ARRAY, _msgpack_enc(meta))

    elif isinstance(obj, BaseException):
        return repr(obj)
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")


def _msgpack_ext_hook(code: int, data: bytes) -> Any:
    if code == EXT_CONSTRUCTOR_SINGLE_ARG:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, arg
            return getattr(importlib.import_module(tup[0]), tup[1])(tup[2])
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_POS_ARGS:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, args
            return getattr(importlib.import_module(tup[0]), tup[1])(*tup[2])
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_KW_ARGS:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, args
            return getattr(importlib.import_module(tup[0]), tup[1])(**tup[2])
        except Exception:
            return
    elif code == EXT_METHOD_SINGLE_ARG:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, arg, method
            return getattr(getattr(importlib.import_module(tup[0]), tup[1]), tup[3])(
                tup[2]
            )
        except Exception:
            return
    elif code == EXT_PYDANTIC_V1:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, kwargs
            cls = getattr(importlib.import_module(tup[0]), tup[1])
            try:
                return cls(**tup[2])
            except Exception:
                return cls.construct(**tup[2])
        except Exception:
            # for pydantic objects we can't find/reconstruct
            # let's return the kwargs dict instead
            try:
                return tup[2]
            except NameError:
                return
    elif code == EXT_PYDANTIC_V2:
        try:
            tup = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            # module, name, kwargs, method
            cls = getattr(importlib.import_module(tup[0]), tup[1])
            try:
                return cls(**tup[2])
            except Exception:
                return cls.model_construct(**tup[2])
        except Exception:
            # for pydantic objects we can't find/reconstruct
            # let's return the kwargs dict instead
            try:
                return tup[2]
            except NameError:
                return
    elif code == EXT_NUMPY_ARRAY:
        try:
            import numpy as _np

            dtype_str, shape, order, buf = ormsgpack.unpackb(
                data, ext_hook=_msgpack_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            arr = _np.frombuffer(buf, dtype=_np.dtype(dtype_str))
            return arr.reshape(shape, order=order)
        except Exception:
            return


def _msgpack_ext_hook_to_json(code: int, data: bytes) -> Any:
    if code == EXT_CONSTRUCTOR_SINGLE_ARG:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            if tup[0] == "uuid" and tup[1] == "UUID":
                hex_ = tup[2]
                return (
                    f"{hex_[:8]}-{hex_[8:12]}-{hex_[12:16]}-{hex_[16:20]}-{hex_[20:]}"
                )
            # module, name, arg
            return tup[2]
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_POS_ARGS:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            if tup[0] == "langgraph.types" and tup[1] == "Send":
                from langgraph.types import Send  # type: ignore

                return Send(*tup[2])
            # module, name, args
            return tup[2]
        except Exception:
            return
    elif code == EXT_CONSTRUCTOR_KW_ARGS:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            # module, name, args
            return tup[2]
        except Exception:
            return
    elif code == EXT_METHOD_SINGLE_ARG:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            # module, name, arg, method
            return tup[2]
        except Exception:
            return
    elif code == EXT_PYDANTIC_V1:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            # module, name, kwargs
            return tup[2]
        except Exception:
            # for pydantic objects we can't find/reconstruct
            # let's return the kwargs dict instead
            return
    elif code == EXT_PYDANTIC_V2:
        try:
            tup = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            # module, name, kwargs, method
            return tup[2]
        except Exception:
            return
    elif code == EXT_NUMPY_ARRAY:
        try:
            import numpy as _np

            dtype_str, shape, order, buf = ormsgpack.unpackb(
                data,
                ext_hook=_msgpack_ext_hook_to_json,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
            arr = _np.frombuffer(buf, dtype=_np.dtype(dtype_str))
            return arr.reshape(shape, order=order).tolist()
        except Exception:
            return


class InvalidModuleError(Exception):
    """Exception raised when a module is not in the allowlist."""

    def __init__(self, message: str):
        self.message = message


_option = (
    ormsgpack.OPT_NON_STR_KEYS
    | ormsgpack.OPT_PASSTHROUGH_DATACLASS
    | ormsgpack.OPT_PASSTHROUGH_DATETIME
    | ormsgpack.OPT_PASSTHROUGH_ENUM
    | ormsgpack.OPT_PASSTHROUGH_UUID
)


def _msgpack_enc(data: Any) -> bytes:
    return ormsgpack.packb(data, default=_msgpack_default, option=_option)
