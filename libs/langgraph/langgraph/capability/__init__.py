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

__all__ = [
    "CapabilityContractError",
    "CapabilityError",
    "CapabilityInvocationError",
    "CapabilitySchemaError",
    "CapabilitySpec",
    "CapabilityVersionError",
    "GraphCapability",
    "SemVer",
    "SideEffect",
    "StateBoundary",
    "attach_capability",
    "graph_capability",
    "select_capability_version",
    "validate_capability_spec",
]
