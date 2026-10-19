from __future__ import annotations

import dataclasses
import functools
import logging
import sys
import types
from collections import deque
from enum import Enum
from inspect import isfunction, ismethod
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
    from langgraph.checkpoint.serde._msgpack import (
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
    control_type_sources: list[Any] | None = None,
) -> set[tuple[str, ...]]:
    allowlist = curated_core_allowlist()
    if schemas:
        schemas = [schema for schema in schemas if schema is not None]
    return (
        allowlist
        | collect_allowlist_from_schemas(
            schemas=schemas,
            channels=channels,
        )
        | collect_control_types_from_callables(control_type_sources)
    )


# Control-flow types admitted to the derived allowlist only when the graph's
# own nodes or routers reference them (see collect_control_types_from_callables).
_control_type_classes: tuple[type, ...] | None = None

# Bound on the static scan below: defensive cap so exotic node objects
# (mocks, dynamic attributes) cannot turn compile-time analysis into a loop.
_MAX_SCAN_OBJECTS = 5000


def _get_control_type_classes() -> tuple[type, ...]:
    # Imported lazily: jsonplus takes the same precaution for this cycle.
    global _control_type_classes
    if _control_type_classes is None:
        from langgraph.types import (
            Command,
            Overwrite,
            Send,
        )

        _control_type_classes = (Send, Command, Overwrite)
    return _control_type_classes


def _control_type_key(obj: Any) -> tuple[str, ...] | None:
    if not isinstance(obj, type):
        return None
    for cls in _get_control_type_classes():
        if obj is cls or (issubclass(obj, cls) and obj.__module__ == "langgraph.types"):
            return ("langgraph.types", obj.__name__)
    return None


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _unwrap_to_function(obj: Any) -> Any:
    """Best-effort unwrap of a node/router runnable to its underlying function."""
    if isfunction(obj) or ismethod(obj):
        return obj
    if isinstance(obj, functools.partial):
        return _unwrap_to_function(obj.func)
    for attr in ("func", "afunc"):
        fn = _safe_getattr(obj, attr)
        if fn is not None and (isfunction(fn) or ismethod(fn)):
            return fn
    call = _safe_getattr(obj, "__call__")
    if isfunction(call) or ismethod(call):
        return call
    return None


def _names_touched_by(func: Any) -> set[str]:
    """Names loaded by a function, including nested defs and comprehensions."""
    names: set[str] = set()
    stack: list[Any] = [_safe_getattr(func, "__code__")]
    while stack:
        code = stack.pop()
        if not isinstance(code, types.CodeType):
            continue
        names.update(code.co_names)
        names.update(code.co_freevars)
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return names


def _resolve_control_types(
    names: set[str], namespace: dict[str, Any]
) -> set[tuple[str, ...]]:
    """Resolve which names in a function's namespace are control-flow types.

    Covers ``from langgraph.types import Send`` (direct binding),
    ``import langgraph`` + ``langgraph.types.Send`` / ``langgraph.Send``
    (one- and two-hop module attribute chains), and closure bindings.
    """
    found: set[tuple[str, ...]] = set()
    for name in names:
        obj = namespace.get(name)
        if obj is None:
            continue
        if (key := _control_type_key(obj)) is not None:
            found.add(key)
        elif isinstance(obj, types.ModuleType):
            for inner in names:
                sub = _safe_getattr(obj, inner)
                if sub is None:
                    continue
                if (key := _control_type_key(sub)) is not None:
                    found.add(key)
                elif isinstance(sub, types.ModuleType):
                    for leaf in names:
                        leaf_obj = _safe_getattr(sub, leaf)
                        if (
                            leaf_obj is not None
                            and (key := _control_type_key(leaf_obj)) is not None
                        ):
                            found.add(key)
    return found


def _control_types_referenced_by(func: Any) -> set[tuple[str, ...]]:
    code = _safe_getattr(func, "__code__")
    if not isinstance(code, types.CodeType):
        return set()
    namespaces: list[dict[str, Any]] = []
    if isinstance(globals_ := _safe_getattr(func, "__globals__"), dict):
        namespaces.append(globals_)
    closure_namespace: dict[str, Any] = {}
    cells = _safe_getattr(func, "__closure__") or ()
    for name, cell in zip(code.co_freevars, cells):
        try:
            closure_namespace[name] = cell.cell_contents
        except ValueError:
            # empty cell
            pass
    if closure_namespace:
        namespaces.append(closure_namespace)
    names = _names_touched_by(func)
    found: set[tuple[str, ...]] = set()
    for namespace in namespaces:
        found |= _resolve_control_types(names, namespace)
    return found


def collect_control_types_from_callables(
    sources: list[Any] | None,
) -> set[tuple[str, ...]]:
    """Return control-flow (module, name) keys statically referenced by graph
    callables.

    Accepts node runnables, branch path runnables, node specs, or compiled
    graphs; recurses into subgraphs (``.nodes``), branch writer chains
    (``.writers``), and branch specs (``.path``, reached through bound methods
    like ``Branch._route``).

    This is deliberately an under-approximation of "the graph can emit a
    control type at runtime": static analysis cannot see dynamically
    constructed Sends. Graphs that build control types without referencing
    them lexically need an explicit ``allowed_msgpack_modules`` entry in
    strict mode; the blocked-deserialization warning names the exact key.
    """
    found: set[tuple[str, ...]] = set()
    if not sources:
        return found
    seen: set[int] = set()
    stack: list[Any] = list(sources)
    budget = _MAX_SCAN_OBJECTS
    while stack and budget > 0:
        budget -= 1
        obj = stack.pop()
        if isinstance(obj, (str, bytes, bytearray, type)):
            continue
        try:
            marker = id(obj)
        except Exception:
            continue
        if marker in seen:
            continue
        seen.add(marker)

        # Compiled graphs (CompiledStateGraph et al.) expose their nodes.
        nodes = _safe_getattr(obj, "nodes")
        if isinstance(nodes, dict):
            stack.extend(nodes.values())
        # Node specs keep the main runnable under .runnable (builder-level
        # StateNodeSpec) or .bound (PregelNode); branch specs keep .path.
        for attr in ("runnable", "bound", "path"):
            value = _safe_getattr(obj, attr)
            if value is not None:
                stack.append(value)
        # PregelNode writer chains (branch routers) and nested subgraphs.
        for attr in ("writers", "subgraphs"):
            value = _safe_getattr(obj, attr)
            if value is not None:
                try:
                    stack.extend(value)
                except TypeError:
                    pass
        # Namedtuple specs (BranchSpec) expose their fields via _asdict.
        asdict = _safe_getattr(obj, "_asdict")
        if callable(asdict):
            try:
                stack.extend(v for v in asdict().values() if v is not None)
            except Exception:
                pass

        fn = _unwrap_to_function(obj)
        if fn is not None:
            found |= _control_types_referenced_by(fn)
            # A bound method (e.g. Branch._route) points back at its spec
            # instance, whose .path holds the user's router.
            bound_self = _safe_getattr(fn, "__self__")
            if bound_self is not None and not isinstance(bound_self, type):
                spec_asdict = _safe_getattr(bound_self, "_asdict")
                if callable(spec_asdict):
                    try:
                        stack.extend(v for v in spec_asdict().values() if v is not None)
                    except Exception:
                        pass
                elif isinstance(_safe_getattr(bound_self, "__dict__"), dict):
                    try:
                        stack.extend(
                            v for v in vars(bound_self).values() if v is not None
                        )
                    except Exception:
                        pass
    return found


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
