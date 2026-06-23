"""Graph capabilities: versioned, reusable graphs as packages and services.

LangGraph executes graphs well; capabilities make them *distributable* assets
with stable I/O contracts, semver, and composition boundaries.
"""

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
    "CapabilityContractError",
    "CapabilityError",
    "CapabilityInvocationError",
    "CapabilitySchemaError",
    "CapabilitySpec",
    "CapabilityVersionError",
    "GraphCapability",
    "SemVer",
    "ServiceCapability",
    "ServiceEndpoint",
    "ServiceRunResult",
    "ServiceRunStatus",
    "SideEffect",
    "StateBoundary",
    "attach_capability",
    "attach_service_capability",
    "graph_capability",
    "iter_boundary_events",
    "local_service_invoker",
    "select_capability_version",
    "service_capability",
    "service_capability_from_package",
    "validate_capability_spec",
]
