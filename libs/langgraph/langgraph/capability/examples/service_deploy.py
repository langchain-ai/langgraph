"""Service-mode deploy surface for the research capability.

Production: point ``langgraph.json`` at :func:`build_research_service_graph`
(same implementation core as the package builder; provider bindings would be
injected here in a real deployment).

Consumers call via :class:`~langgraph.capability.service.ServiceCapability`
or ``RemoteGraph``, not by importing internal nodes.
"""

from __future__ import annotations

from typing import Any

from langgraph.capability.examples.research import (
    RESEARCH_CAPABILITY,
    RESEARCH_SPEC,
    build_research_graph,
)
from langgraph.capability.service import (
    ServiceEndpoint,
    local_service_invoker,
    service_capability,
    service_capability_from_package,
)

# Graph id / assistant id used in deploy configs and service clients.
RESEARCH_SERVICE_GRAPH_ID = "research"


def build_research_service_graph() -> Any:
    """Deploy entrypoint (provider side): same logic as package, service wiring.

    In a real org service you would bind secrets/tools/models here, while
    keeping the public I/O contract identical to the package capability.
    """
    return build_research_graph(prefix="svc", max_sources_default=3)


def research_service_capability_for_tests(
    *,
    prefix: str = "svc",
) -> Any:
    """Hermetic service capability (package-backed invoker) for unit tests."""
    return service_capability_from_package(
        RESEARCH_CAPABILITY,
        ServiceEndpoint(
            assistant_id=RESEARCH_SERVICE_GRAPH_ID,
            graph_id=RESEARCH_SERVICE_GRAPH_ID,
            version_label=RESEARCH_SPEC.version,
        ),
        invoker=local_service_invoker(RESEARCH_CAPABILITY, prefix=prefix),
    )


def research_service_capability_remote(
    *,
    url: str,
    api_key: str | None = None,
    assistant_id: str = RESEARCH_SERVICE_GRAPH_ID,
) -> Any:
    """Live service handle (requires a running LangGraph Server deployment)."""
    return service_capability(
        RESEARCH_SPEC,
        ServiceEndpoint(
            url=url,
            assistant_id=assistant_id,
            api_key=api_key,
            graph_id=RESEARCH_SERVICE_GRAPH_ID,
            version_label=RESEARCH_SPEC.version,
        ),
    )


# Minimal langgraph.json fragment (documented for deploy authors).
LANGGRAPH_JSON_EXAMPLE = {
    "graphs": {
        RESEARCH_SERVICE_GRAPH_ID: (
            "langgraph.capability.examples.service_deploy:build_research_service_graph"
        )
    }
}
