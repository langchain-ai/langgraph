from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

import langgraph_sdk
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


# The tests above compare implementation signatures, which is what
# `inspect.signature` resolves to. Type checkers never see those: for a method
# with `@overload` stubs they consider only the stubs, so a parameter missing
# from the stubs is unusable in typed code even though it works at runtime. The
# tests below therefore read the stubs themselves.
#
# The stubs are parsed with `ast` rather than `typing.get_overloads` because the
# latter needs Python 3.11 and this package supports 3.10.

_SDK_DIR = pathlib.Path(langgraph_sdk.__file__).parent

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

# Modules under _async/ and _sync/ that declare @overload stubs.
OVERLOADED_MODULES = ("assistants", "runs", "threads")

# Runs parameters that are deliberately restricted to one `thread_id` variant,
# and so are legitimately absent from the other variant's stubs.
# `checkpoint`/`checkpoint_id`/`multitask_strategy` act on a thread the caller
# supplies, so they cannot apply to a stateless run; `on_completion` decides the
# fate of the thread the server creates for a stateless run, so it cannot apply
# when the caller supplies the thread.
STATEFUL_ONLY_PARAMS = frozenset({"checkpoint", "checkpoint_id", "multitask_strategy"})
STATELESS_ONLY_PARAMS = frozenset({"on_completion"})


def _parse_client_module(kind: str, module: str) -> ast.Module:
    return ast.parse((_SDK_DIR / kind / f"{module}.py").read_text())


def _is_overload(fn: FunctionNode) -> bool:
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _params(fn: FunctionNode) -> set[str]:
    args = fn.args
    return {
        arg.arg
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        if arg.arg != "self"
    }


def _thread_variant(fn: FunctionNode) -> str:
    """Report which kind of run a stub describes, per its `thread_id` type."""
    for arg in (*fn.args.posonlyargs, *fn.args.args):
        if arg.arg == "thread_id":
            annotation = arg.annotation
            if isinstance(annotation, ast.Constant) and annotation.value is None:
                return "stateless"
    return "stateful"


def _overload_groups(
    tree: ast.Module,
) -> dict[tuple[str, str], tuple[list[FunctionNode], FunctionNode]]:
    """Map (class, method) to its overload stubs and its implementation."""
    groups: dict[tuple[str, str], tuple[list[FunctionNode], FunctionNode]] = {}
    for cls in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        defs: dict[str, list[FunctionNode]] = {}
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(fn.name, []).append(fn)
        for name, fns in defs.items():
            stubs = [fn for fn in fns if _is_overload(fn)]
            impls = [fn for fn in fns if not _is_overload(fn)]
            if not stubs:
                continue
            assert len(impls) == 1, (
                f"{cls.name}.{name} has {len(impls)} implementations, expected 1"
            )
            groups[(cls.name, name)] = (stubs, impls[0])
    return groups


@pytest.mark.parametrize("kind", ["_async", "_sync"])
@pytest.mark.parametrize("module", OVERLOADED_MODULES)
def test_overload_stubs_cover_the_implementation(kind: str, module: str) -> None:
    """Every parameter must be reachable through at least one stub."""
    groups = _overload_groups(_parse_client_module(kind, module))
    assert groups, f"no overloaded methods found in {kind}/{module}.py"

    for (cls_name, method), (stubs, impl) in groups.items():
        impl_params = _params(impl)
        declared: set[str] = set()
        for stub in stubs:
            stub_params = _params(stub)
            assert stub_params <= impl_params, (
                f"{cls_name}.{method} has an @overload stub declaring parameters "
                f"the implementation does not accept: "
                f"{sorted(stub_params - impl_params)}"
            )
            declared |= stub_params
        assert declared == impl_params, (
            f"{cls_name}.{method} accepts parameters that no @overload stub "
            f"declares, so type checkers reject them even though they work at "
            f"runtime: {sorted(impl_params - declared)}"
        )


@pytest.mark.parametrize("module", OVERLOADED_MODULES)
def test_overload_stubs_match_between_sync_and_async(module: str) -> None:
    """The hand-maintained sync stubs must mirror the async ones."""
    async_groups = _overload_groups(_parse_client_module("_async", module))
    sync_groups = _overload_groups(_parse_client_module("_sync", module))

    def keyed(groups: dict) -> dict:
        return {
            (cls_name.removeprefix("Sync"), method): value
            for (cls_name, method), value in groups.items()
        }

    async_by_key = keyed(async_groups)
    sync_by_key = keyed(sync_groups)
    assert async_by_key.keys() == sync_by_key.keys(), (
        f"Overloaded methods differ in {module}.py: "
        f"async-only={sorted(async_by_key.keys() - sync_by_key.keys())}, "
        f"sync-only={sorted(sync_by_key.keys() - async_by_key.keys())}"
    )

    for (cls_name, method), (async_stubs, _) in async_by_key.items():
        sync_stubs, _ = sync_by_key[(cls_name, method)]
        assert len(async_stubs) == len(sync_stubs), (
            f"{cls_name}.{method} has {len(async_stubs)} async stubs but "
            f"{len(sync_stubs)} sync stubs"
        )
        paired = zip(async_stubs, sync_stubs, strict=True)
        for index, (async_stub, sync_stub) in enumerate(paired):
            async_params = _params(async_stub)
            sync_params = _params(sync_stub)
            assert async_params == sync_params, (
                f"Overload {index} of {cls_name}.{method} differs between the "
                f"async and sync clients: "
                f"async-only={sorted(async_params - sync_params)}, "
                f"sync-only={sorted(sync_params - async_params)}"
            )


@pytest.mark.parametrize("kind", ["_async", "_sync"])
def test_runs_overload_stubs_match_their_thread_variant(kind: str) -> None:
    """Each runs stub must declare exactly what its `thread_id` variant allows."""
    groups = _overload_groups(_parse_client_module(kind, "runs"))

    for (cls_name, method), (stubs, impl) in groups.items():
        impl_params = _params(impl)
        for stub in stubs:
            variant = _thread_variant(stub)
            omitted = (
                STATEFUL_ONLY_PARAMS
                if variant == "stateless"
                else STATELESS_ONLY_PARAMS
            )
            expected = impl_params - omitted
            actual = _params(stub)
            assert actual == expected, (
                f"The {variant} @overload stub for {cls_name}.{method} has "
                f"drifted from the implementation: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
