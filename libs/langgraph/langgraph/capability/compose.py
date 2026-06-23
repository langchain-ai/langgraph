"""Ergonomic parent-graph composition for capabilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.capability.catalog import CapabilityCatalog
from langgraph.capability.config_ref import resolve_capability_ref
from langgraph.capability.errors import CapabilityContractError
from langgraph.capability.package import GraphCapability, attach_capability
from langgraph.capability.service import ServiceCapability, attach_service_capability

CapabilityMode = Literal["package", "service", "auto"]


def add_capability_node(
    parent: Any,
    node_name: str,
    capability: GraphCapability[Any, Any, Any]
    | ServiceCapability[Any, Any, Any]
    | str,
    /,
    *,
    mode: CapabilityMode = "auto",
    version: str = "*",
    catalog: CapabilityCatalog | None = None,
    public_params: Mapping[str, Any] | None = None,
    input_mapper: Callable[[Any], Any] | None = None,
    output_mapper: Callable[[Any], Any] | None = None,
    default_url: str | None = None,
    default_api_key: str | None = None,
    **add_node_kwargs: Any,
) -> Any:
    """Add a capability as a parent node (package, service, catalog id, or config ref).

    Parameters
    ----------
    capability:
        A :class:`GraphCapability`, :class:`ServiceCapability`, capability id
        (resolved via ``catalog``), or config ref string (``python:``, ``service:``,
        ``catalog:``).
    mode:
        ``package`` / ``service`` force delivery; ``auto`` picks from the object
        or catalog preference (package first, then service).
    """
    resolved = _resolve_capability_handle(
        capability,
        mode=mode,
        version=version,
        catalog=catalog,
        default_url=default_url,
        default_api_key=default_api_key,
    )

    if isinstance(resolved, ServiceCapability):
        return attach_service_capability(
            parent,
            node_name,
            resolved,
            public_params=public_params,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            **add_node_kwargs,
        )
    if isinstance(resolved, GraphCapability):
        return attach_capability(
            parent,
            node_name,
            resolved,
            public_params=public_params,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            **add_node_kwargs,
        )
    raise CapabilityContractError(
        f"add_capability_node could not attach {type(resolved)!r} as a capability node"
    )


def _resolve_capability_handle(
    capability: GraphCapability[Any, Any, Any]
    | ServiceCapability[Any, Any, Any]
    | str,
    *,
    mode: CapabilityMode,
    version: str,
    catalog: CapabilityCatalog | None,
    default_url: str | None,
    default_api_key: str | None,
) -> GraphCapability[Any, Any, Any] | ServiceCapability[Any, Any, Any]:
    if isinstance(capability, (GraphCapability, ServiceCapability)):
        if mode == "package" and isinstance(capability, ServiceCapability):
            raise CapabilityContractError(
                "mode='package' but a ServiceCapability was provided"
            )
        if mode == "service" and isinstance(capability, GraphCapability):
            raise CapabilityContractError(
                "mode='service' but a GraphCapability was provided"
            )
        return capability

    if not isinstance(capability, str):
        raise CapabilityContractError(
            "capability must be GraphCapability, ServiceCapability, id, or config ref"
        )

    # Config ref schemes
    if capability.startswith(("python:", "service:", "catalog:")):
        obj = resolve_capability_ref(
            capability,
            catalog=catalog,
            default_url=default_url,
            default_api_key=default_api_key,
        )
        if isinstance(obj, (GraphCapability, ServiceCapability)):
            return obj
        if callable(obj) and catalog is None:
            # python: builder — not directly attachable without wrapping
            raise CapabilityContractError(
                f"Ref {capability!r} resolved to a builder/callable; "
                "use a GraphCapability export or catalog: ref instead"
            )
        raise CapabilityContractError(
            f"Ref {capability!r} did not resolve to a capability handle"
        )

    # Bare capability id via catalog
    if catalog is None:
        raise CapabilityContractError(
            f"Resolving capability id {capability!r} requires a catalog"
        )
    prefer: Literal["package", "service", "both"]
    if mode == "service":
        prefer = "service"
        return catalog.get_service(capability, version)
    if mode == "package":
        return catalog.get_package(capability, version)
    # auto
    try:
        return catalog.get_package(capability, version)
    except CapabilityContractError:
        return catalog.get_service(capability, version)
