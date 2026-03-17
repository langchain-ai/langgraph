from __future__ import annotations

import ctypes
import dataclasses
import json
import subprocess
from copy import deepcopy
from pathlib import Path
import threading
from typing import Any, Callable, get_args, get_origin


class PyRustEngine:
    def __init__(self) -> None:
        self._lib = _load_rust_lib()
        self._engine = self._lib.rc_engine_new()
        if not self._engine:
            raise RuntimeError("failed to create rust engine")

    def __del__(self) -> None:
        engine = getattr(self, "_engine", None)
        if engine:
            self._lib.rc_engine_free(engine)
            self._engine = None

    def add_async_channel(self, name: str) -> None:
        self._call_status(self._lib.rc_add_async_channel, name.encode())

    def publish_obj(self, channel: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self._call_status(self._lib.rc_publish_json, channel.encode(), payload)

    def wait_any_of_obj(self, any_of_payload: Any) -> Any:
        payload = json.dumps(any_of_payload, ensure_ascii=False).encode()
        raw = self._consume_json_ptr(self._lib.rc_wait_any_of_json(self._engine, payload))
        if not raw.get("ok"):
            raise ValueError(raw.get("error", "rust wait_any_of failed"))
        return raw["event"]

    def wait_channel(self, channel: str, n: int) -> Any:
        return self.wait_any_of_obj({"conditions": [{"kind": "channel", "channel": channel, "n": n}]})

    def wait_timer(self, seconds: float) -> Any:
        return self.wait_any_of_obj({"conditions": [{"kind": "timer", "seconds": seconds}]})

    def start_stream(self, stream_mode: str | None) -> None:
        encoded = stream_mode.encode() if stream_mode is not None else None
        self._call_status(self._lib.rc_start_stream, encoded)

    def receive_stream_obj(self) -> Any | None:
        raw = self._consume_json_ptr(self._lib.rc_receive_stream_json(self._engine))
        if not raw.get("ok"):
            raise ValueError(raw.get("error", "rust receive_stream failed"))
        if not raw.get("has_event", False):
            return None
        return raw.get("event")

    def send_custom_stream_event_obj(self, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self._call_status(self._lib.rc_send_custom_stream_event, payload)

    def close_stream(self) -> None:
        self._call_status(self._lib.rc_close_stream)

    def run_graph_py(
        self,
        entry_point: str,
        finish_point: str,
        initial_state: Any,
        callback: Callable[[str, Any, Any], dict[str, Any]],
        stream_mode: str | None = None,
    ) -> Any:
        shared_state = initial_state
        shared_state_lock = threading.Lock()
        state_type = type(initial_state)
        use_shared_state = dataclasses.is_dataclass(initial_state)
        initial_state_json = json.dumps(_to_jsonable(shared_state), ensure_ascii=False).encode()
        callback_c = _make_node_callback(
            callback,
            state_type,
            use_shared_state,
            shared_state,
            shared_state_lock,
        )
        stream_mode_encoded = stream_mode.encode() if stream_mode is not None else None
        out = self._consume_json_ptr(
            self._lib.rc_run_graph_json(
                self._engine,
                entry_point.encode(),
                finish_point.encode(),
                initial_state_json,
                initial_state_json,
                stream_mode_encoded,
                ctypes.c_ulong(0),
                callback_c,
            )
        )
        if not out.get("ok"):
            raise ValueError(out.get("error", "rust run_graph failed"))
        if use_shared_state:
            return shared_state
        return _coerce_for_type(out["state"], state_type)

    def _call_status(self, func: Any, *args: Any) -> None:
        raw = self._consume_json_ptr(func(self._engine, *args))
        if not raw.get("ok"):
            raise ValueError(raw.get("error", "rust call failed"))

    def _consume_json_ptr(self, ptr: ctypes.c_void_p) -> dict[str, Any]:
        if not ptr:
            raise RuntimeError("rust returned null string pointer")
        try:
            text = ctypes.cast(ptr, ctypes.c_char_p).value
            if text is None:
                raise RuntimeError("rust returned empty string pointer")
            return json.loads(text.decode())
        finally:
            self._lib.rc_string_free(ptr)


def _make_node_callback(
    callback: Callable[[str, Any, Any], dict[str, Any]],
    state_type: type[Any],
    use_shared_state: bool,
    shared_state: Any,
    shared_state_lock: threading.Lock,
) -> ctypes.CFUNCTYPE:  # type: ignore[type-arg]
    cb_type = ctypes.CFUNCTYPE(
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    )
    libc = ctypes.CDLL(None)
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p

    @cb_type
    def _callback(
        _user_data: int,
        node_ptr: bytes,
        arg_ptr: bytes,
        state_ptr: bytes,
    ) -> ctypes.c_void_p:
        try:
            node = node_ptr.decode()
            arg = json.loads(arg_ptr.decode())
            if use_shared_state:
                with shared_state_lock:
                    before = deepcopy(_to_jsonable(shared_state))
                    result = callback(node, arg, shared_state)
                    state_after_call = shared_state
            else:
                state_raw = json.loads(state_ptr.decode())
                state_snapshot = _coerce_for_type(state_raw, state_type)
                before = deepcopy(_to_jsonable(state_snapshot))
                result = callback(node, arg, state_snapshot)
                state_after_call = state_snapshot
            if "suspend" in result:
                envelope = {"ok": True, "suspend": result["suspend"]}
            else:
                update = result.get("update")
                if update is not None:
                    update_json = _to_jsonable(update)
                    if update_json == before:
                        update = None
                        update_json = None
                    else:
                        if use_shared_state:
                            _apply_update_to_state(shared_state, update)
                after = _to_jsonable(state_after_call)
                if update is None and after != before:
                    update_json = after
                elif update is None:
                    update_json = None
                else:
                    update_json = _to_jsonable(update)
                sends = []
                for item in result.get("sends", []):
                    if not isinstance(item, dict):
                        continue
                    sends.append(
                        {
                            "node": item.get("node"),
                            "arg": _to_jsonable(item.get("arg")),
                        }
                    )
                envelope = {
                    "ok": True,
                    "payload": {
                        "update": update_json,
                        "sends": sends,
                    },
                }
        except Exception as exc:  # noqa: BLE001
            envelope = {"ok": False, "error": f"python callback failed: {exc}"}
        return _malloc_c_string(json.dumps(envelope, ensure_ascii=False).encode(), libc)

    return _callback


def _malloc_c_string(payload: bytes, libc: Any) -> ctypes.c_void_p:
    size = len(payload) + 1
    ptr = libc.malloc(size)
    if not ptr:
        return ctypes.c_void_p(0)
    ctypes.memmove(ptr, payload, len(payload))
    ctypes.memset(ctypes.c_void_p(ptr + len(payload)), 0, 1)
    return ptr


def _load_rust_lib() -> ctypes.CDLL:
    env = Path.cwd()
    root = _find_repo_root(env)
    rust_core = root / "rust-core"
    lib_path = _resolve_lib_path(rust_core)
    if not lib_path.exists():
        subprocess.run(["cargo", "build"], cwd=rust_core, check=True)
    lib = ctypes.CDLL(str(lib_path))
    _configure_signatures(lib)
    return lib


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "rust-core").exists() and (candidate / "saf-python-sdk").exists():
            return candidate
    here = Path(__file__).resolve()
    return here.parents[3]


def _resolve_lib_path(rust_core: Path) -> Path:
    if (rust_core / "target" / "debug" / "liblanggraph_rust_core.dylib").exists():
        return rust_core / "target" / "debug" / "liblanggraph_rust_core.dylib"
    if (rust_core / "target" / "debug" / "liblanggraph_rust_core.so").exists():
        return rust_core / "target" / "debug" / "liblanggraph_rust_core.so"
    if (rust_core / "target" / "debug" / "langgraph_rust_core.dll").exists():
        return rust_core / "target" / "debug" / "langgraph_rust_core.dll"
    return rust_core / "target" / "debug" / "liblanggraph_rust_core.dylib"


def _configure_signatures(lib: ctypes.CDLL) -> None:
    cb_type = ctypes.CFUNCTYPE(
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    )
    lib.rc_engine_new.argtypes = []
    lib.rc_engine_new.restype = ctypes.c_void_p
    lib.rc_engine_free.argtypes = [ctypes.c_void_p]
    lib.rc_engine_free.restype = None
    lib.rc_string_free.argtypes = [ctypes.c_void_p]
    lib.rc_string_free.restype = None
    lib.rc_add_async_channel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.rc_add_async_channel.restype = ctypes.c_void_p
    lib.rc_publish_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.rc_publish_json.restype = ctypes.c_void_p
    lib.rc_wait_any_of_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.rc_wait_any_of_json.restype = ctypes.c_void_p
    lib.rc_start_stream.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.rc_start_stream.restype = ctypes.c_void_p
    lib.rc_receive_stream_json.argtypes = [ctypes.c_void_p]
    lib.rc_receive_stream_json.restype = ctypes.c_void_p
    lib.rc_send_custom_stream_event.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.rc_send_custom_stream_event.restype = ctypes.c_void_p
    lib.rc_close_stream.argtypes = [ctypes.c_void_p]
    lib.rc_close_stream.restype = ctypes.c_void_p
    lib.rc_run_graph_json.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        cb_type,
    ]
    lib.rc_run_graph_json.restype = ctypes.c_void_p


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def _coerce_for_type(value: Any, typ: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is not None:
        if origin in (list, tuple, set):
            item_type = args[0] if args else Any
            items = [_coerce_for_type(v, item_type) for v in value]
            if origin is tuple:
                return tuple(items)
            if origin is set:
                return set(items)
            return items
        if origin is dict:
            value_type = args[1] if len(args) == 2 else Any
            return {k: _coerce_for_type(v, value_type) for k, v in value.items()}
    if isinstance(typ, type):
        if dataclasses.is_dataclass(typ):
            kwargs = {}
            for field in dataclasses.fields(typ):
                kwargs[field.name] = _coerce_for_type(value.get(field.name), field.type)
            return typ(**kwargs)
        model_validate = getattr(typ, "model_validate", None)
        if callable(model_validate):
            return model_validate(value)
    return value


def _apply_update_to_state(state: Any, update: Any) -> None:
    if update is None:
        return
    if isinstance(state, dict):
        if isinstance(update, dict):
            state.update(update)
            return
        if dataclasses.is_dataclass(update):
            state.update(_to_jsonable(update))
            return
        return
    if dataclasses.is_dataclass(state):
        if dataclasses.is_dataclass(update):
            for field in dataclasses.fields(state):
                setattr(state, field.name, getattr(update, field.name))
            return
        if isinstance(update, dict):
            for key, val in update.items():
                setattr(state, key, val)
