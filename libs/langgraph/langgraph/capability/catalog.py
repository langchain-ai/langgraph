"""Capability catalog: register, list, and resolve versioned capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.capability.contract import CapabilitySpec, select_capability_version
from langgraph.capability.errors import CapabilityContractError, CapabilityVersionError
from langgraph.capability.package import GraphCapability
from langgraph.capability.service import ServiceCapability

Delivery = Literal["package", "service", "both"]


@dataclass
class CatalogEntry:
    """One registered implementation of a capability (package and/or service)."""

    spec: CapabilitySpec[Any, Any, Any]
    package: GraphCapability[Any, Any, Any] | None = None
    service: ServiceCapability[Any, Any, Any] | None = None
    tags: tuple[str, ...] = ()
    owner: str = ""

    def __post_init__(self) -> None:
        if self.package is None and self.service is None:
            raise CapabilityContractError(
                "CatalogEntry requires at least one of package or service"
            )
        if self.package is not None and self.package.spec.capability_id != self.spec.capability_id:
            raise CapabilityContractError("package spec capability_id mismatch")
        if self.service is not None and self.service.spec.capability_id != self.spec.capability_id:
            raise CapabilityContractError("service spec capability_id mismatch")
        if not self.owner:
            self.owner = self.spec.owner

    @property
    def capability_id(self) -> str:
        return self.spec.capability_id

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def deliveries(self) -> list[Delivery]:
        if self.package and self.service:
            return ["both"]
        if self.package:
            return ["package"]
        return ["service"]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_metadata(),
            "owner": self.owner,
            "tags": list(self.tags),
            "deliveries": self.deliveries,
            "has_package": self.package is not None,
            "has_service": self.service is not None,
        }


@dataclass
class CapabilityCatalog:
    """In-process registry of capabilities (org catalog / app wiring)."""

    _entries: list[CatalogEntry] = field(default_factory=list)

    def register(self, entry: CatalogEntry) -> CatalogEntry:
        self._entries.append(entry)
        return entry

    def register_package(
        self,
        capability: GraphCapability[Any, Any, Any],
        /,
        *,
        tags: Sequence[str] = (),
        owner: str = "",
        service: ServiceCapability[Any, Any, Any] | None = None,
    ) -> CatalogEntry:
        return self.register(
            CatalogEntry(
                spec=capability.spec,
                package=capability,
                service=service,
                tags=tuple(tags),
                owner=owner or capability.spec.owner,
            )
        )

    def register_service(
        self,
        capability: ServiceCapability[Any, Any, Any],
        /,
        *,
        tags: Sequence[str] = (),
        owner: str = "",
        package: GraphCapability[Any, Any, Any] | None = None,
    ) -> CatalogEntry:
        return self.register(
            CatalogEntry(
                spec=capability.spec,
                package=package,
                service=capability,
                tags=tuple(tags),
                owner=owner or capability.spec.owner,
            )
        )

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self._entries)

    def list(
        self,
        *,
        capability_id: str | None = None,
        tag: str | None = None,
        delivery: Delivery | None = None,
    ) -> list[CatalogEntry]:
        out: list[CatalogEntry] = []
        for entry in self._entries:
            if capability_id and entry.capability_id != capability_id:
                continue
            if tag and tag not in entry.tags:
                continue
            if delivery:
                d = entry.deliveries[0]
                if delivery == "both":
                    if d != "both":
                        continue
                elif delivery == "package" and entry.package is None:
                    continue
                elif delivery == "service" and entry.service is None:
                    continue
            out.append(entry)
        return out

    def resolve(
        self,
        capability_id: str,
        version: str = "*",
        /,
        *,
        prefer: Delivery = "package",
    ) -> CatalogEntry:
        """Resolve the best entry for id + semver request."""
        candidates = self.list(capability_id=capability_id)
        if not candidates:
            raise CapabilityVersionError(
                f"No capability registered with id {capability_id!r}"
            )
        specs = [e.spec for e in candidates]
        chosen_spec = select_capability_version(specs, capability_id, version)
        matches = [e for e in candidates if e.spec.version == chosen_spec.version]
        if not matches:
            raise CapabilityVersionError(
                f"Internal error resolving {capability_id!r}@{version}"
            )
        if prefer == "service":
            for e in matches:
                if e.service is not None:
                    return e
        if prefer == "package":
            for e in matches:
                if e.package is not None:
                    return e
        if prefer == "both":
            for e in matches:
                if e.package is not None and e.service is not None:
                    return e
        return matches[0]

    def get_package(
        self, capability_id: str, version: str = "*", /
    ) -> GraphCapability[Any, Any, Any]:
        entry = self.resolve(capability_id, version, prefer="package")
        if entry.package is None:
            raise CapabilityContractError(
                f"{capability_id!r}@{entry.version} has no package delivery"
            )
        return entry.package

    def get_service(
        self, capability_id: str, version: str = "*", /
    ) -> ServiceCapability[Any, Any, Any]:
        entry = self.resolve(capability_id, version, prefer="service")
        if entry.service is None:
            raise CapabilityContractError(
                f"{capability_id!r}@{entry.version} has no service delivery"
            )
        return entry.service

    def to_summary(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]


def default_example_catalog() -> CapabilityCatalog:
    """Catalog populated with reference examples (docs/tests)."""
    from langgraph.capability.examples.research import RESEARCH_CAPABILITY
    from langgraph.capability.examples.review import REVIEW_CAPABILITY
    from langgraph.capability.examples.service_deploy import (
        research_service_capability_for_tests,
    )

    cat = CapabilityCatalog()
    cat.register_package(
        RESEARCH_CAPABILITY,
        tags=("reference", "research"),
        service=research_service_capability_for_tests(),
    )
    cat.register_package(REVIEW_CAPABILITY, tags=("reference", "review"))
    return cat
