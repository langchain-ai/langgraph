from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from langgraph_sdk.client import (
    AssistantsClient,
    CronClient,
    RunsClient,
    StoreClient,
    SyncAssistantsClient,
    SyncCronClient,
    SyncRunsClient,
    SyncStoreClient,
    SyncThreadsClient,
    ThreadsClient,
)


def _public_methods(cls) -> dict[str, object]:
    methods: dict[str, object] = {}
    # Use the raw class dict to avoid runtime wrappers from plugins/decorators
    for name, member in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(member):
            methods[name] = member
    return methods


def _strip_self(sig: inspect.Signature) -> inspect.Signature:
    params = list(sig.parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return sig.replace(parameters=params)


def _normalize_return_annotation(ann: object) -> str:
    s = str(ann)
    s = re.sub(r"\s+", "", s)
    s = s.replace("typing.", "").replace("collections.abc.", "")
    s = re.sub(r"AsyncGenerator\[([^,\]]+)(?:,[^\]]*)?\]", r"Iterator[\1]", s)
    s = re.sub(r"Generator\[([^,\]]+)(?:,[^\]]*)?\]", r"Iterator[\1]", s)
    s = re.sub(r"AsyncIterator\[(.+)\]", r"Iterator[\1]", s)
    s = re.sub(r"AsyncIterable\[(.+)\]", r"Iterable[\1]", s)
    # Normalize Async/Sync class prefixes so AsyncFoo and SyncFoo both compare as Foo.
    s = re.sub(r"\bAsync([A-Z])", r"\1", s)
    s = re.sub(r"\bSync([A-Z])", r"\1", s)
    return s


ASYNC_ONLY_METHODS: dict[str, set[str]] = {}


@pytest.mark.parametrize(
    "async_cls,sync_cls",
    [
        (AssistantsClient, SyncAssistantsClient),
        (ThreadsClient, SyncThreadsClient),
        (RunsClient, SyncRunsClient),
        (CronClient, SyncCronClient),
        (StoreClient, SyncStoreClient),
    ],
)
def test_sync_api_matches_async(async_cls, sync_cls):
    async_methods = _public_methods(async_cls)
    sync_methods = _public_methods(sync_cls)

    allowlist = ASYNC_ONLY_METHODS.get(async_cls.__name__, set())
    async_method_names = set(async_methods.keys()) - allowlist

    # Method name parity (modulo the async-only allowlist).
    assert sync_methods.keys() == async_method_names, (
        f"Method sets differ: async-only={async_method_names - set(sync_methods)}, sync-only={set(sync_methods) - async_method_names}"
    )

    for name in async_method_names:
        async_fn = async_methods[name]
        sync_fn = sync_methods[name]

        # Use inspect.signature for parameter names (robust across versions)
        async_sig = _strip_self(inspect.signature(async_fn))  # type: ignore
        sync_sig = _strip_self(inspect.signature(sync_fn))  # type: ignore

        a_names = list(async_sig.parameters.keys())
        s_names = list(sync_sig.parameters.keys())

        assert set(a_names) == set(s_names), (
            f"Parameter names differ for {async_cls.__name__}.{name}: "
            f"async={a_names}, sync={s_names}"
        )

        # Compare default presence and parameter kinds (with some tolerance)
        a_params = async_sig.parameters
        s_params = sync_sig.parameters

        def kinds_compatible(
            akind: inspect._ParameterKind, skind: inspect._ParameterKind
        ) -> bool:
            if akind == skind:
                return True
            return {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            } == {akind, skind}

        for pname in set(a_names) & set(s_names):
            apar = a_params[pname]
            spar = s_params[pname]
            assert kinds_compatible(apar.kind, spar.kind), (
                f"Parameter kind mismatch for {async_cls.__name__}.{name}.{pname}: "
                f"async={apar.kind}, sync={spar.kind}"
            )
            assert (apar.default is inspect._empty) == (
                spar.default is inspect._empty
            ), (
                f"Default presence mismatch for {async_cls.__name__}.{name}.{pname}: "
                f"async_has_default={apar.default is not inspect._empty}, "
                f"sync_has_default={spar.default is not inspect._empty}"
            )

        # Return annotations must match or be iterator-equivalent
        a_ret = _normalize_return_annotation(async_sig.return_annotation)
        s_ret = _normalize_return_annotation(sync_sig.return_annotation)
        assert a_ret == s_ret, (
            f"Return annotation mismatch for {async_cls.__name__}.{name}: "
            f"async={a_ret}, sync={s_ret}"
        )


# Stateless overloads (`thread_id: None`) leave out the thread-only parameters,
# and stateful ones (`thread_id: str`) leave out `on_completion`, which only
# applies to stateless runs.
_RUN_OVERLOAD_OMISSIONS = {
    "None": {"checkpoint", "checkpoint_id", "multitask_strategy"},
    "str": {"on_completion"},
}


@pytest.mark.parametrize("client_cls", [RunsClient, SyncRunsClient])
def test_run_overloads_accept_implementation_params(client_cls) -> None:
    tree = ast.parse(Path(inspect.getfile(client_cls)).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == client_cls.__name__
    )
    defs = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"stream", "create", "wait"}
    ]
    impl_params = {
        node.name: {arg.arg for arg in node.args.kwonlyargs}
        for node in defs
        if not node.decorator_list
    }
    gaps = []
    for stub in defs:
        if not stub.decorator_list:
            continue
        thread_id = next(arg for arg in stub.args.args if arg.arg == "thread_id")
        assert thread_id.annotation is not None
        kind = ast.unparse(thread_id.annotation)
        expected = impl_params[stub.name] - _RUN_OVERLOAD_OMISSIONS[kind]
        if missing := expected - {arg.arg for arg in stub.args.kwonlyargs}:
            gaps.append(f"{stub.name}(thread_id: {kind}) line {stub.lineno}: {missing}")

    assert not gaps, gaps
