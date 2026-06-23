"""Service-mode graph capabilities: black-box remote delivery.

Uses existing LangGraph Server / ``RemoteGraph`` patterns. Consumers see only
capability id, semver, I/O schemas, and boundary status—not internal nodes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from langgraph.capability.contract import CapabilitySpec
from langgraph.capability.errors import (
    CapabilityContractError,
    CapabilityError,
    CapabilityInvocationError,
    CapabilitySchemaError,
    CapabilityVersionError,
)
from langgraph.capability.package import (
    GraphCapability,
    _validate_against_schema,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")


class ServiceRunStatus(str, Enum):
    """Coarse boundary-level run status (service mode is intentionally opaque)."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """How to reach a deployed capability implementation."""

    url: str | None = None
    """LangGraph Server base URL (optional if a graph/client is injected)."""

    assistant_id: str | None = None
    """Assistant or graph id on the server (capability implementation handle)."""

    api_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    graph_id: str | None = None
    """Optional deploy graph id (often same as assistant_id / capability name)."""

    version_label: str | None = None
    """Advertised semver or deploy tag for this endpoint (e.g. ``1.0.0`` or ``1``)."""


@dataclass
class ServiceRunResult(Generic[OutputT]):
    """Boundary result from a service capability invocation."""

    output: OutputT | None
    status: ServiceRunStatus
    run_id: str
    capability_id: str
    version: str
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceCapability(Generic[InputT, OutputT, ParamsT]):
    """Black-box remote capability with the same contract as package mode.

    Provider injects secrets/tools/policy on the server. Callers pass only
    capability input and optional **public** params (never provider bindings).
    """

    spec: CapabilitySpec[InputT, OutputT, ParamsT]
    endpoint: ServiceEndpoint
    validate_io: bool = True
    # Injectable transports for tests / custom clients (no new deps).
    invoker: Callable[[InputT, dict[str, Any]], Any] | None = None
    async_invoker: Callable[[InputT, dict[str, Any]], Any] | None = None
    remote_graph: Any | None = None
    """Optional pre-built ``RemoteGraph`` (or compatible invokable)."""

    def __post_init__(self) -> None:
        if (
            self.invoker is None
            and self.remote_graph is None
            and not (self.endpoint.url and self.endpoint.assistant_id)
        ):
            # Allow construct-only for metadata/catalog; invoke will fail clearly.
            pass
        if self.endpoint.version_label and not self.spec.supports_version_request(
            self.endpoint.version_label
        ):
            # Endpoint may advertise a major line (``1``) while spec is exact semver.
            try:
                from langgraph.capability.contract import SemVer

                SemVer.parse(self.endpoint.version_label)
                if self.endpoint.version_label != self.spec.version:
                    pass  # exact semver mismatch is ok if caller pins endpoint explicitly
            except CapabilityVersionError:
                if not self.spec.supports_version_request(self.endpoint.version_label):
                    raise CapabilityVersionError(
                        f"Endpoint version_label {self.endpoint.version_label!r} "
                        f"does not satisfy capability {self.spec.capability_id!r} "
                        f"@{self.spec.version}"
                    )

    @property
    def capability_id(self) -> str:
        return self.spec.capability_id

    @property
    def version(self) -> str:
        return self.spec.version

    def _ensure_remote_graph(self) -> Any:
        if self.remote_graph is not None:
            return self.remote_graph
        if not self.endpoint.url or not self.endpoint.assistant_id:
            raise CapabilityContractError(
                "Service capability requires remote_graph, invoker, or "
                "endpoint.url + endpoint.assistant_id"
            )
        from langgraph.pregel.remote import RemoteGraph

        self.remote_graph = RemoteGraph(
            self.endpoint.assistant_id,
            url=self.endpoint.url,
            api_key=self.endpoint.api_key,
            headers=dict(self.endpoint.headers),
            name=self.spec.capability_id,
        )
        return self.remote_graph

    def _config(self, public_params: Mapping[str, Any] | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "metadata": {
                "capability_id": self.spec.capability_id,
                "capability_version": self.spec.version,
                "delivery": "service",
            }
        }
        if public_params:
            cfg["configurable"] = dict(public_params)
        return cfg

    def invoke(
        self,
        input: InputT,
        /,
        *,
        public_params: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> OutputT:
        result = self.invoke_with_status(
            input, public_params=public_params, config=config
        )
        if result.status is not ServiceRunStatus.SUCCEEDED or result.output is None:
            raise CapabilityInvocationError(
                result.error_message
                or f"Service capability {self.spec.capability_id!r} did not succeed",
                capability_id=self.spec.capability_id,
                version=self.spec.version,
                run_id=result.run_id,
            )
        return result.output

    def invoke_with_status(
        self,
        input: InputT,
        /,
        *,
        public_params: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> ServiceRunResult[OutputT]:
        run_id = str(uuid.uuid4())
        if self.validate_io:
            _validate_against_schema("input", self.spec.input_schema, input)
        if self.spec.public_params_schema is not None and public_params is not None:
            _validate_against_schema(
                "public_params", self.spec.public_params_schema, public_params
            )

        merged_config = self._config(public_params)
        if config:
            # Shallow-merge caller config under our capability metadata.
            meta = {**merged_config.get("metadata", {}), **config.get("metadata", {})}
            merged_config = {**config, **merged_config, "metadata": meta}
            if "configurable" in config or "configurable" in merged_config:
                merged_config["configurable"] = {
                    **config.get("configurable", {}),
                    **merged_config.get("configurable", {}),
                }

        try:
            if self.invoker is not None:
                raw = self.invoker(input, merged_config)
            else:
                graph = self._ensure_remote_graph()
                raw = graph.invoke(input, merged_config)
        except CapabilityError:
            raise
        except Exception as exc:
            return ServiceRunResult(
                output=None,
                status=ServiceRunStatus.FAILED,
                run_id=run_id,
                capability_id=self.spec.capability_id,
                version=self.spec.version,
                error_message=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )

        if self.validate_io:
            try:
                _validate_against_schema("output", self.spec.output_schema, raw)
            except CapabilitySchemaError as exc:
                return ServiceRunResult(
                    output=None,
                    status=ServiceRunStatus.FAILED,
                    run_id=run_id,
                    capability_id=self.spec.capability_id,
                    version=self.spec.version,
                    error_message=str(exc),
                )

        return ServiceRunResult(
            output=cast(OutputT, raw),
            status=ServiceRunStatus.SUCCEEDED,
            run_id=run_id,
            capability_id=self.spec.capability_id,
            version=self.spec.version,
            metadata={
                "endpoint_assistant_id": self.endpoint.assistant_id,
                "endpoint_version_label": self.endpoint.version_label,
            },
        )

    async def ainvoke(
        self,
        input: InputT,
        /,
        *,
        public_params: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> OutputT:
        run_id = str(uuid.uuid4())
        if self.validate_io:
            _validate_against_schema("input", self.spec.input_schema, input)
        merged_config = self._config(public_params)
        if config:
            meta = {**merged_config.get("metadata", {}), **config.get("metadata", {})}
            merged_config = {**config, **merged_config, "metadata": meta}

        try:
            if self.async_invoker is not None:
                raw = await self.async_invoker(input, merged_config)
            elif self.invoker is not None:
                raw = self.invoker(input, merged_config)
            else:
                graph = self._ensure_remote_graph()
                if hasattr(graph, "ainvoke"):
                    raw = await graph.ainvoke(input, merged_config)
                else:
                    raw = graph.invoke(input, merged_config)
        except Exception as exc:
            raise CapabilityInvocationError(
                f"Service capability {self.spec.capability_id!r} failed",
                capability_id=self.spec.capability_id,
                version=self.spec.version,
                run_id=run_id,
                cause=exc,
            ) from exc

        if self.validate_io:
            _validate_against_schema("output", self.spec.output_schema, raw)
        return cast(OutputT, raw)

    def as_node(self) -> Callable[[Any], Any]:
        """Return a parent-graph node function that calls this service capability.

        Expects parent state (or prior mapper) to already match the capability input
        schema, or use :func:`attach_service_capability` with mappers.
        """

        def _node(state: Any) -> Any:
            return self.invoke(state)

        _node.__name__ = self.spec.capability_id.replace(".", "_")
        return _node

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.spec.to_metadata(),
            "delivery": "service",
            "endpoint": {
                "url": self.endpoint.url,
                "assistant_id": self.endpoint.assistant_id,
                "graph_id": self.endpoint.graph_id,
                "version_label": self.endpoint.version_label,
            },
        }


def service_capability(
    spec: CapabilitySpec[InputT, OutputT, ParamsT],
    endpoint: ServiceEndpoint,
    /,
    *,
    invoker: Callable[[InputT, dict[str, Any]], Any] | None = None,
    async_invoker: Callable[[InputT, dict[str, Any]], Any] | None = None,
    remote_graph: Any | None = None,
    validate_io: bool = True,
) -> ServiceCapability[InputT, OutputT, ParamsT]:
    """Create a service-mode capability bound to a deploy endpoint or test invoker."""
    return ServiceCapability(
        spec=spec,
        endpoint=endpoint,
        invoker=invoker,
        async_invoker=async_invoker,
        remote_graph=remote_graph,
        validate_io=validate_io,
    )


def service_capability_from_package(
    package_cap: GraphCapability[InputT, OutputT, ParamsT],
    endpoint: ServiceEndpoint,
    /,
    **kwargs: Any,
) -> ServiceCapability[InputT, OutputT, ParamsT]:
    """Wrap the same contract as a service handle (typical: package is source of truth)."""
    return service_capability(package_cap.spec, endpoint, **kwargs)


def local_service_invoker(
    package_cap: GraphCapability[InputT, OutputT, ParamsT],
    /,
    **build_params: Any,
) -> Callable[[InputT, dict[str, Any]], Any]:
    """Test/dev helper: simulate a service by invoking the package implementation.

    Production services deploy the graph; this keeps contract tests hermetic without
    a live server.
    """

    def _invoke(input: InputT, config: dict[str, Any]) -> Any:
        public = (config or {}).get("configurable", {})
        merged = {**build_params, **public}
        return package_cap.invoke(input, **merged)

    return _invoke


def attach_service_capability(
    parent: Any,
    node_name: str,
    capability: ServiceCapability[Any, Any, Any],
    /,
    *,
    public_params: Mapping[str, Any] | None = None,
    input_mapper: Callable[[Any], Any] | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
    **add_node_kwargs: Any,
) -> Any:
    """Attach a service capability as a parent node (remote black-box step)."""

    def _node(state: Any) -> Any:
        payload = input_mapper(state) if input_mapper is not None else state
        result = capability.invoke(payload, public_params=public_params)
        if output_mapper is not None:
            return output_mapper(result)
        return result

    _node.__name__ = node_name
    return parent.add_node(node_name, _node, **add_node_kwargs)


def iter_boundary_events(
    result: ServiceRunResult[Any],
) -> Iterator[dict[str, Any]]:
    """Yield coarse boundary observability events (not internal node traces)."""
    yield {
        "type": "capability_run",
        "capability_id": result.capability_id,
        "version": result.version,
        "run_id": result.run_id,
        "status": result.status.value,
        "error_message": result.error_message,
        "metadata": result.metadata,
    }
