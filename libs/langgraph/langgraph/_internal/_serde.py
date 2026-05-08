from __future__ import annotations

import dataclasses
import logging
import sys
import types
from collections import deque
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core import messages as lc_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel
from typing_extensions import NotRequired, Required, is_typeddict

try:
    from langgraph.checkpoint.serde._msgpack import (  # noqa: F401
        STRICT_MSGPACK_ENABLED,
    )
except ImportError:
    STRICT_MSGPACK_ENABLED = False

_warned_allowlist_unsupported = False

logger = logging.getLogger(__name__)


def _supports_checkpointer_allowlist() -> bool:
    return hasattr(BaseCheckpointSaver, "with_allowlist")


_SUPPORTS_ALLOWLIST = _supports_checkpointer_allowlist()


def apply_checkpointer_allowlist(
    checkpointer: Any, allowlist: set[tuple[str, ...]] | None
) -> Any:
    if not checkpointer or allowlist is None or checkpointer in (True, False):
        return checkpointer
    if not _SUPPORTS_ALLOWLIST:
        global _warned_allowlist_unsupported
        if not _warned_allowlist_unsupported:
            logger.warning(
                "Checkpointer does not support with_allowlist; strict msgpack "
                "allowlist will be skipped."
            )
            _warned_allowlist_unsupported = True
        return checkpointer
    return checkpointer.with_allowlist(allowlist)


def curated_core_allowlist() -> set[tuple[str, ...]]:
    allowlist: set[tuple[str, ...]] = set()
    for name in (
        "BaseMessage",
        "BaseMessageChunk",
        "HumanMessage",
        "HumanMessageChunk",
        "AIMessage",
        "AIMessageChunk",
        "SystemMessage",
        "SystemMessageChunk",
        "ChatMessage",
        "ChatMessageChunk",
        "ToolMessage",
        "ToolMessageChunk",
        "FunctionMessage",
        "FunctionMessageChunk",
        "RemoveMessage",
    ):
        cls = getattr(lc_messages, name, None)
        if cls is None:
            continue
        allowlist.add((cls.__module__, cls.__name__))

    return allowlist


def build_serde_allowlist(
    *,
    schemas: list[type[Any]] | None = None,
    channels: dict[str, Any] | None = None,
) -> set[tuple[str, ...]]:
    allowlist = curated_core_allowlist()
    if schemas:
        schemas = [schema for schema in schemas if schema is not None]
    return allowlist | collect_allowlist_from_schemas(
        schemas=schemas,
        channels=channels,
    )


def collect_allowlist_from_schemas(
    *,
    schemas: list[type[Any]] | None = None,
    channels: dict[str, Any] | None = None,
) -> set[tuple[str, ...]]:
    allowlist: set[tuple[str, ...]] = set()
    seen: set[Any] = set()
    seen_ids: set[int] = set()

    if schemas:
        for schema in schemas:
            _collect_from_type(schema, allowlist, seen, seen_ids)

    if channels:
        for channel in channels.values():
            value_type = getattr(channel, "ValueType", None)
            if value_type is not None:
                _collect_from_type(value_type, allowlist, seen, seen_ids)
            update_type = getattr(channel, "UpdateType", None)
            if update_type is not None:
                _collect_from_type(update_type, allowlist, seen, seen_ids)

    return allowlist


def _collect_from_type(
    typ: Any,
    allowlist: set[tuple[str, ...]],
    seen: set[Any],
    seen_ids: set[int],
) -> None:
    if _already_seen(typ, seen, seen_ids):
        return

    if typ is Any or typ is None:
        return

    if typ is Literal:
        return

    if isinstance(typ, types.UnionType):
        for arg in typ.__args__:
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return

    origin = get_origin(typ)
    if origin is Union:
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return
    if origin is Annotated or origin in (Required, NotRequired):
        args = get_args(typ)
        if args:
            _collect_from_type(args[0], allowlist, seen, seen_ids)
        return

    if origin is Literal:
        return

    if origin in (list, set, tuple, dict, deque, frozenset):
        for arg in get_args(typ):
            _collect_from_type(arg, allowlist, seen, seen_ids)
        return

    if hasattr(typ, "__supertype__"):
        _collect_from_type(typ.__supertype__, allowlist, seen, seen_ids)
        return

    if is_typeddict(typ):
        for field_type in _safe_get_type_hints(typ).values():
            _collect_from_type(field_type, allowlist, seen, seen_ids)
        return

    if _is_pydantic_model(typ):
        allowlist.add((typ.__module__, typ.__name__))
        field_types = _safe_get_type_hints(typ)
        if field_types:
            for field_type in field_types.values():
                _collect_from_type(field_type, allowlist, seen, seen_ids)
        else:
            for field_type in _pydantic_field_types(typ):
                _collect_from_type(field_type, allowlist, seen, seen_ids)
        return

    if dataclasses.is_dataclass(typ):
        if typ_name := getattr(typ, "__name__", None):
            allowlist.add((typ.__module__, typ_name))
        field_types = _safe_get_type_hints(typ)
        if field_types:
            for field_type in field_types.values():
                _collect_from_type(field_type, allowlist, seen, seen_ids)
        else:
            for field in dataclasses.fields(typ):
                _collect_from_type(field.type, allowlist, seen, seen_ids)
        return

    if isinstance(typ, type) and issubclass(typ, Enum):
        allowlist.add((typ.__module__, typ.__name__))
        return


def _already_seen(typ: Any, seen: set[Any], seen_ids: set[int]) -> bool:
    try:
        if typ in seen:
            return True
        seen.add(typ)
        return False
    except TypeError:
        typ_id = id(typ)
        if typ_id in seen_ids:
            return True
        seen_ids.add(typ_id)
        return False


def _safe_get_type_hints(typ: Any) -> dict[str, Any]:
    try:
        module = sys.modules.get(getattr(typ, "__module__", ""))
        globalns = module.__dict__ if module else None
        localns = dict(vars(typ)) if hasattr(typ, "__dict__") else None
        return get_type_hints(
            typ, globalns=globalns, localns=localns, include_extras=True
        )
    except Exception:
        return {}


def _is_pydantic_model(typ: Any) -> bool:
    if not isinstance(typ, type):
        return False
    if issubclass(typ, BaseModel):
        return True
    try:
        from pydantic.v1 import BaseModel as BaseModelV1
    except Exception:
        return False
    return issubclass(typ, BaseModelV1)


def _pydantic_field_types(typ: type[Any]) -> list[Any]:
    if hasattr(typ, "model_fields"):
        return [
            field.annotation
            for field in typ.model_fields.values()
            if getattr(field, "annotation", None) is not None
        ]
    if hasattr(typ, "__fields__"):
        return [
            field.outer_type_
            for field in typ.__fields__.values()
            if getattr(field, "outer_type_", None) is not None
        ]
    return []
