"""Package-mode graph capabilities: local, installable, composable as subgraphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from langgraph.capability.contract import Builder, CapabilitySpec
from langgraph.capability.errors import (
    CapabilityContractError,
    CapabilityInvocationError,
    CapabilitySchemaError,
)
from langgraph.pregel.protocol import PregelProtocol
from langgraph.typing import ContextT

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _required_typed_dict_keys(schema: type[Any]) -> list[str]:
    """Return required keys for a TypedDict (honors total=False and NotRequired)."""
    annotations = getattr(schema, "__annotations__", None)
    if not annotations:
        return []
    required_keys = getattr(schema, "__required_keys__", None)
    if required_keys is not None:
        # Prefer runtime TypedDict metadata when present and non-empty or total=False.
        optional_keys = getattr(schema, "__optional_keys__", frozenset())
        if required_keys or optional_keys or getattr(schema, "__total__", True) is False:
            return list(required_keys)
    if getattr(schema, "__total__", True) is False:
        return []
    # Strip NotRequired[...] / Required[...] wrappers when metadata is missing.
    required: list[str] = []
    for key, anno in annotations.items():
        origin = getattr(anno, "__origin__", None)
        name = getattr(origin, "__name__", "") or getattr(anno, "_name", "")
        if name == "NotRequired":
            continue
        required.append(key)
    return required


def _validate_against_schema(label: str, schema: type[Any], value: Any) -> None:
    """Best-effort validation for TypedDict/dataclass/pydantic/simple types."""
    if schema is Any:
        return
    # Pydantic v2 model
    model_validate = getattr(schema, "model_validate", None)
    if callable(model_validate):
        try:
            model_validate(value)
            return
        except Exception as exc:
            raise CapabilitySchemaError(
                f"{label} failed schema validation for {getattr(schema, '__name__', schema)}: {exc}"
            ) from exc
    # TypedDict / dict-shaped
    annotations = getattr(schema, "__annotations__", None)
    if annotations is not None and _is_mapping(value):
        missing = [k for k in _required_typed_dict_keys(schema) if k not in value]
        if missing:
            raise CapabilitySchemaError(
                f"{label} missing required keys for "
                f"{getattr(schema, '__name__', schema)}: {missing}"
            )
        return
    # Concrete type check when schema is a runtime type
    if isinstance(schema, type) and schema not in (dict, Mapping) and not annotations:
        if not isinstance(value, schema):
            raise CapabilitySchemaError(
                f"{label} expected {schema.__name__}, got {type(value).__name__}"
            )


@dataclass
class GraphCapability(Generic[InputT, OutputT, ParamsT]):
    """A reusable graph capability delivered as an in-process package.

    Consumers depend on :attr:`spec` only. Internals (node names, private state)
    are not part of the public contract.
    """

    spec: CapabilitySpec[InputT, OutputT, ParamsT]
    builder: Builder
    """Builds a compiled/buildable graph given public params only."""

    default_params: Mapping[str, Any] = field(default_factory=dict)
    validate_io: bool = True

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise CapabilityContractError("builder must be callable")

    @property
    def capability_id(self) -> str:
        return self.spec.capability_id

    @property
    def version(self) -> str:
        return self.spec.version

    def build(self, **public_params: Any) -> Any:
        """Build the underlying graph using public params only (no secrets/provider bindings)."""
        merged = {**self.default_params, **public_params}
        if self.spec.public_params_schema is not None and self.validate_io:
            _validate_against_schema(
                "public_params", self.spec.public_params_schema, merged
            )
        try:
            return self.builder(**merged)
        except TypeError as exc:
            raise CapabilityContractError(
                f"builder for {self.spec.capability_id!r} rejected params: {exc}"
            ) from exc

    def as_node(self, **public_params: Any) -> Any:
        """Return a graph/runnable suitable for ``StateGraph.add_node``."""
        graph = self.build(**public_params)
        return graph

    def invoke(
        self,
        input: InputT,
        config: Any | None = None,
        /,
        **public_params: Any,
    ) -> OutputT:
        """Invoke the capability at its I/O boundary (standalone or in tests)."""
        if self.validate_io:
            _validate_against_schema("input", self.spec.input_schema, input)
        graph = self.build(**public_params)
        if not hasattr(graph, "invoke"):
            raise CapabilityContractError(
                f"builder for {self.spec.capability_id!r} must return an invokable graph"
            )
        try:
            result = graph.invoke(input, config)
        except Exception as exc:
            raise CapabilityInvocationError(
                f"Capability {self.spec.capability_id!r}@{self.spec.version} failed",
                capability_id=self.spec.capability_id,
                version=self.spec.version,
                cause=exc,
            ) from exc
        if self.validate_io:
            _validate_against_schema("output", self.spec.output_schema, result)
        return cast(OutputT, result)

    async def ainvoke(
        self,
        input: InputT,
        config: Any | None = None,
        /,
        **public_params: Any,
    ) -> OutputT:
        if self.validate_io:
            _validate_against_schema("input", self.spec.input_schema, input)
        graph = self.build(**public_params)
        if not hasattr(graph, "ainvoke"):
            raise CapabilityContractError(
                f"builder for {self.spec.capability_id!r} must return an async-invokable graph"
            )
        try:
            result = await graph.ainvoke(input, config)
        except Exception as exc:
            raise CapabilityInvocationError(
                f"Capability {self.spec.capability_id!r}@{self.spec.version} failed",
                capability_id=self.spec.capability_id,
                version=self.spec.version,
                cause=exc,
            ) from exc
        if self.validate_io:
            _validate_against_schema("output", self.spec.output_schema, result)
        return cast(OutputT, result)

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.spec.to_metadata(),
            "delivery": "package",
            "entrypoint": getattr(self.builder, "__qualname__", repr(self.builder)),
        }


def graph_capability(
    spec: CapabilitySpec[InputT, OutputT, ParamsT],
    builder: Builder,
    *,
    default_params: Mapping[str, Any] | None = None,
    validate_io: bool = True,
) -> GraphCapability[InputT, OutputT, ParamsT]:
    """Create a package-mode capability from a contract and builder entrypoint."""
    return GraphCapability(
        spec=spec,
        builder=builder,
        default_params=dict(default_params or {}),
        validate_io=validate_io,
    )


def attach_capability(
    parent: Any,
    node_name: str,
    capability: GraphCapability[Any, Any, Any],
    /,
    *,
    public_params: Mapping[str, Any] | None = None,
    input_mapper: Callable[[Any], Any] | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
    **add_node_kwargs: Any,
) -> Any:
    """Attach a package capability as a parent graph node.

    By default the compiled child graph is added directly (isolated subgraph).
    Provide ``input_mapper`` / ``output_mapper`` for explicit mapped boundaries
    when parent state shape differs from capability I/O schemas.
    """
    child = capability.as_node(**dict(public_params or {}))
    if input_mapper is None and output_mapper is None:
        return parent.add_node(node_name, child, **add_node_kwargs)

    def _mapped_node(state: Any) -> Any:
        payload = input_mapper(state) if input_mapper is not None else state
        if capability.validate_io:
            _validate_against_schema("input", capability.spec.input_schema, payload)
        if hasattr(child, "invoke"):
            result = child.invoke(payload)
        else:
            result = child(payload)
        if capability.validate_io:
            _validate_against_schema("output", capability.spec.output_schema, result)
        if output_mapper is not None:
            return output_mapper(result)
        return result

    _mapped_node.__name__ = node_name
    return parent.add_node(node_name, _mapped_node, **add_node_kwargs)


# Re-export for typing convenience in examples/tests
PregelLike = PregelProtocol[Any, ContextT, Any, Any]
