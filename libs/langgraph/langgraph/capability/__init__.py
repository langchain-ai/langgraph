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
from langgraph.capability.parity import (
    DEFAULT_PARITY_DISCLAIMER,
    ParityReport,
    assert_io_compatible,
    compare_capability_parity,
    is_breaking_schema_change,
)
from langgraph.capability.progress import (
    CapabilityProgressEvent,
    ProgressPhase,
    emit_run_progress,
    iter_progress_dicts,
    progress_events_from_run_result,
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
from langgraph.capability.versioning import (
    ServiceVersionPolicy,
    VersionWindow,
    bind_service_to_policy,
    default_n_minus_one_policy,
)

__all__ = [
    "CONFIG_REF_EXAMPLES",
    "DEFAULT_PARITY_DISCLAIMER",
    "CapabilityCatalog",
    "CapabilityContractError",
    "CapabilityError",
    "CapabilityInvocationError",
    "CapabilityProgressEvent",
    "CapabilityRef",
    "CapabilitySchemaError",
    "CapabilitySpec",
    "CapabilityVersionError",
    "CatalogEntry",
    "GraphCapability",
    "ParityReport",
    "ProgressPhase",
    "SemVer",
    "ServiceCapability",
    "ServiceEndpoint",
    "ServiceRunResult",
    "ServiceRunStatus",
    "ServiceVersionPolicy",
    "SideEffect",
    "StateBoundary",
    "VersionWindow",
    "add_capability_node",
    "assert_io_compatible",
    "attach_capability",
    "attach_service_capability",
    "bind_service_to_policy",
    "compare_capability_parity",
    "default_example_catalog",
    "default_n_minus_one_policy",
    "emit_run_progress",
    "graph_capability",
    "is_breaking_schema_change",
    "iter_boundary_events",
    "iter_progress_dicts",
    "local_service_invoker",
    "parse_capability_ref",
    "progress_events_from_run_result",
    "resolve_capability_ref",
    "resolve_python_ref",
    "select_capability_version",
    "service_capability",
    "service_capability_from_package",
    "validate_capability_spec",
]
