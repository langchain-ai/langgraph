"""Graph capabilities: versioned, reusable graphs as packages and services.

LangGraph executes graphs well; capabilities make them *distributable* assets
with stable I/O contracts, semver, and composition boundaries.
"""

from langgraph.capability.catalog import (
    CapabilityCatalog,
    CatalogEntry,
    default_example_catalog,
)
from langgraph.capability.compose import add_capability_node
from langgraph.capability.config_ref import (
    CONFIG_REF_EXAMPLES,
    CapabilityRef,
    parse_capability_ref,
    resolve_capability_ref,
    resolve_python_ref,
)
from langgraph.capability.contract import (
    CapabilitySpec,
    SemVer,
    SideEffect,
    StateBoundary,
    select_capability_version,
    validate_capability_spec,
)
from langgraph.capability.errors import (
    CapabilityContractError,
    CapabilityError,
    CapabilityInvocationError,
    CapabilitySchemaError,
    CapabilityVersionError,
)
from langgraph.capability.package import (
    GraphCapability,
    attach_capability,
    graph_capability,
)
from langgraph.capability.service import (
    ServiceCapability,
    ServiceEndpoint,
    ServiceRunResult,
    ServiceRunStatus,
    attach_service_capability,
    iter_boundary_events,
    local_service_invoker,
    service_capability,
    service_capability_from_package,
)

__all__ = [
    "CONFIG_REF_EXAMPLES",
    "CapabilityCatalog",
    "CapabilityContractError",
    "CapabilityError",
    "CapabilityInvocationError",
    "CapabilityRef",
    "CapabilitySchemaError",
    "CapabilitySpec",
    "CapabilityVersionError",
    "CatalogEntry",
    "GraphCapability",
    "SemVer",
    "ServiceCapability",
    "ServiceEndpoint",
    "ServiceRunResult",
    "ServiceRunStatus",
    "SideEffect",
    "StateBoundary",
    "add_capability_node",
    "attach_capability",
    "attach_service_capability",
    "default_example_catalog",
    "graph_capability",
    "iter_boundary_events",
    "local_service_invoker",
    "parse_capability_ref",
    "resolve_capability_ref",
    "resolve_python_ref",
    "select_capability_version",
    "service_capability",
    "service_capability_from_package",
    "validate_capability_spec",
]
