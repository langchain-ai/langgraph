"""Load configured graphs and run langgraph.graph.analysis against them.

This module is deliberately separate from cli.py's `dev`/`up` commands: it
never starts a server and never invokes user code beyond importing the
module that defines each graph, so it is safe to run against a project
without a database, API keys, or any other runtime dependency configured.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class GraphLoadResult:
    graph_id: str
    spec: str
    graph: Any | None
    builder: Any | None
    checkpointer_known: bool
    checkpointer: Any | None
    error: str | None
    skipped_reason: str | None


def _import_from_path(file_path: pathlib.Path, attr_name: str) -> Any:
    module_name = f"_langgraph_lint_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    if not hasattr(module, attr_name):
        raise AttributeError(f"module {file_path} has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def load_graph_object(spec: str, base_dir: pathlib.Path) -> Any:
    """Resolve a `path/to/file.py:attr` or `dotted.module:attr` spec.

    Mirrors the "graphs" value syntax documented in langgraph.json (see
    `GraphDef`/`Config.graphs` in schemas.py): a string is always
    `<path-or-module>:<attribute name>`. File paths are resolved relative
    to `base_dir` (the directory containing the config file), matching
    the resolution used for local dependencies and for the Docker-build
    graph-path remap (`_assemble_local_deps`/`_update_graph_paths` in
    config.py) -- not the process's current working directory.
    """
    if ":" not in spec:
        raise ValueError(f"graph spec '{spec}' is missing the ':<attribute>' suffix")
    path_part, attr_name = spec.rsplit(":", 1)

    candidate = (base_dir / path_part).resolve()
    if candidate.suffix == ".py" and candidate.exists():
        return _import_from_path(candidate, attr_name)
    if path_part.endswith(".py"):
        raise FileNotFoundError(f"graph file not found: {candidate}")

    module = importlib.import_module(path_part)
    if not hasattr(module, attr_name):
        raise AttributeError(f"module '{path_part}' has no attribute '{attr_name}'")
    return getattr(module, attr_name)


def resolve_graph(graph_id: str, spec: str, base_dir: pathlib.Path) -> GraphLoadResult:
    try:
        obj = load_graph_object(spec, base_dir)
    except Exception as exc:  # surfaced to the user as a lint error, not raised
        return GraphLoadResult(
            graph_id=graph_id,
            spec=spec,
            graph=None,
            builder=None,
            checkpointer_known=False,
            checkpointer=None,
            error=f"{type(exc).__name__}: {exc}",
            skipped_reason=None,
        )

    # Compiled graph (Pregel / CompiledStateGraph): has both the original
    # StateGraph builder and the checkpointer it was actually compiled with.
    if hasattr(obj, "builder") and hasattr(obj, "checkpointer"):
        return GraphLoadResult(
            graph_id=graph_id,
            spec=spec,
            graph=obj,
            builder=obj.builder,
            checkpointer_known=True,
            checkpointer=obj.checkpointer,
            error=None,
            skipped_reason=None,
        )

    # Uncompiled StateGraph: has nodes/edges/branches directly, but no
    # checkpointer has been chosen yet.
    if hasattr(obj, "nodes") and hasattr(obj, "edges") and hasattr(obj, "branches"):
        return GraphLoadResult(
            graph_id=graph_id,
            spec=spec,
            graph=obj,
            builder=obj,
            checkpointer_known=False,
            checkpointer=None,
            error=None,
            skipped_reason=None,
        )

    if callable(obj):
        return GraphLoadResult(
            graph_id=graph_id,
            spec=spec,
            graph=obj,
            builder=None,
            checkpointer_known=False,
            checkpointer=None,
            error=None,
            skipped_reason=(
                "graph is a factory/context-manager callable; lint does not "
                "invoke it (that would execute user side effects), so its "
                "structure cannot be statically analyzed"
            ),
        )

    return GraphLoadResult(
        graph_id=graph_id,
        spec=spec,
        graph=obj,
        builder=None,
        checkpointer_known=False,
        checkpointer=None,
        error=None,
        skipped_reason=f"object of type {type(obj).__name__} is not a recognized graph",
    )


def prepare_import_path(base_dir: pathlib.Path, dependencies: list[str]) -> None:
    """Add base_dir and local dependency dirs to sys.path.

    Mirrors the sys.path setup `dev` performs, but rooted at the config
    file's directory rather than the process cwd (see `load_graph_object`).
    """
    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))
    for dep in dependencies:
        dep_path = base_dir / dep
        if dep_path.is_dir() and dep_path.exists() and str(dep_path) not in sys.path:
            sys.path.append(str(dep_path))
