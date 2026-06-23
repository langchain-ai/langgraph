"""Config / deploy references for capabilities (python: and service: schemes)."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.capability.catalog import CapabilityCatalog
from langgraph.capability.errors import CapabilityContractError
from langgraph.capability.package import GraphCapability
from langgraph.capability.service import ServiceCapability, ServiceEndpoint, service_capability

RefKind = Literal["python", "service", "catalog"]

_PYTHON_RE = re.compile(
    r"^python:(?P<module>[\w.]+):(?P<attr>[\w.]+)$"
)
_SERVICE_RE = re.compile(
    r"^service:(?P<id>[\w.-]+)(?:@(?P<version>[\w.*-]+))?(?:\?(?P<qs>.+))?$"
)
_CATALOG_RE = re.compile(
    r"^catalog:(?P<id>[\w.-]+)(?:@(?P<version>[\w.*-]+))?(?::(?P<delivery>package|service))?$"
)


@dataclass(frozen=True)
class CapabilityRef:
    """Parsed reference from config (langgraph.json, app config, catalogs)."""

    kind: RefKind
    raw: str
    module: str | None = None
    attr: str | None = None
    capability_id: str | None = None
    version: str | None = None
    delivery: Literal["package", "service"] | None = None
    query: Mapping[str, str] = field(default_factory=dict)


def parse_capability_ref(ref: str) -> CapabilityRef:
    """Parse ``python:module:attr``, ``service:id@version``, or ``catalog:id@version:package``."""
    ref = ref.strip()
    m = _PYTHON_RE.match(ref)
    if m:
        return CapabilityRef(
            kind="python",
            raw=ref,
            module=m.group("module"),
            attr=m.group("attr"),
            query={},
        )
    m = _SERVICE_RE.match(ref)
    if m:
        qs_raw = m.group("qs") or ""
        query = {}
        if qs_raw:
            for part in qs_raw.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    query[k] = v
        return CapabilityRef(
            kind="service",
            raw=ref,
            capability_id=m.group("id"),
            version=m.group("version") or "*",
            query=query,
        )
    m = _CATALOG_RE.match(ref)
    if m:
        return CapabilityRef(
            kind="catalog",
            raw=ref,
            capability_id=m.group("id"),
            version=m.group("version") or "*",
            delivery=m.group("delivery") or "package",  # type: ignore[arg-type]
            query={},
        )
    raise CapabilityContractError(
        f"Unsupported capability ref {ref!r}; expected python:, service:, or catalog:"
    )


def _resolve_attr(module_name: str, attr_path: str) -> Any:
    mod = importlib.import_module(module_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_python_ref(ref: CapabilityRef | str) -> Any:
    """Import and return the attribute (builder, capability, or compiled graph)."""
    parsed = parse_capability_ref(ref) if isinstance(ref, str) else ref
    if parsed.kind != "python":
        raise CapabilityContractError(f"Not a python: ref: {parsed.raw}")
    assert parsed.module and parsed.attr
    return _resolve_attr(parsed.module, parsed.attr)


def resolve_service_ref(
    ref: CapabilityRef | str,
    /,
    *,
    default_url: str | None = None,
    default_api_key: str | None = None,
    spec_loader: Callable[[str, str], Any] | None = None,
) -> ServiceCapability[Any, Any, Any]:
    """Build a :class:`ServiceCapability` from ``service:id@version?url=...&assistant_id=...``."""
    parsed = parse_capability_ref(ref) if isinstance(ref, str) else ref
    if parsed.kind != "service":
        raise CapabilityContractError(f"Not a service: ref: {parsed.raw}")
    assert parsed.capability_id
    version = parsed.version or "*"
    q = parsed.query or {}
    url = q.get("url", default_url)
    assistant_id = q.get("assistant_id", parsed.capability_id)
    api_key = q.get("api_key", default_api_key)
    graph_id = q.get("graph_id", assistant_id)

    if spec_loader is None:
        raise CapabilityContractError(
            "resolve_service_ref requires spec_loader(capability_id, version) "
            "or use resolve_capability_ref with a catalog"
        )
    spec = spec_loader(parsed.capability_id, version)
    return service_capability(
        spec,
        ServiceEndpoint(
            url=url,
            assistant_id=assistant_id,
            api_key=api_key,
            graph_id=graph_id,
            version_label=version if version not in {"*", "latest"} else None,
        ),
    )


def resolve_catalog_ref(
    ref: CapabilityRef | str,
    catalog: CapabilityCatalog,
) -> GraphCapability[Any, Any, Any] | ServiceCapability[Any, Any, Any]:
    parsed = parse_capability_ref(ref) if isinstance(ref, str) else ref
    if parsed.kind != "catalog":
        raise CapabilityContractError(f"Not a catalog: ref: {parsed.raw}")
    assert parsed.capability_id
    version = parsed.version or "*"
    delivery = parsed.delivery or "package"
    if delivery == "service":
        return catalog.get_service(parsed.capability_id, version)
    return catalog.get_package(parsed.capability_id, version)


def resolve_capability_ref(
    ref: str,
    /,
    *,
    catalog: CapabilityCatalog | None = None,
    default_url: str | None = None,
    default_api_key: str | None = None,
    spec_loader: Callable[[str, str], Any] | None = None,
) -> Any:
    """Resolve any supported ref to a capability, builder, or graph object."""
    parsed = parse_capability_ref(ref)
    if parsed.kind == "python":
        return resolve_python_ref(parsed)
    if parsed.kind == "catalog":
        if catalog is None:
            raise CapabilityContractError("catalog: refs require a CapabilityCatalog")
        return resolve_catalog_ref(parsed, catalog)
    if parsed.kind == "service":
        if catalog is not None and spec_loader is None:
            def _loader(cid: str, ver: str) -> Any:
                entry = catalog.resolve(cid, ver, prefer="service")
                return entry.spec

            spec_loader = _loader
        return resolve_service_ref(
            parsed,
            default_url=default_url,
            default_api_key=default_api_key,
            spec_loader=spec_loader,
        )
    raise CapabilityContractError(f"Unknown ref kind for {ref!r}")


# Example fragments suitable for docs / langgraph.json comments.
CONFIG_REF_EXAMPLES: dict[str, str] = {
    "python_builder": (
        "python:langgraph.capability.examples.research:build_research_graph"
    ),
    "python_capability": (
        "python:langgraph.capability.examples.research:RESEARCH_CAPABILITY"
    ),
    "service_remote": "service:langgraph.research@1?url=http://localhost:2024&assistant_id=research",
    "catalog_package": "catalog:langgraph.research@1:package",
    "catalog_service": "catalog:langgraph.research@1.0.0:service",
}
