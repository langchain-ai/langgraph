"""Package vs service parity notes and compatibility checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.capability.contract import CapabilitySpec, SemVer
from langgraph.capability.errors import CapabilityContractError
from langgraph.capability.package import GraphCapability
from langgraph.capability.service import ServiceCapability

ParityLevel = Literal["identical_io", "compatible_major", "divergent", "unknown"]


# Documented non-parity: honest differences between delivery modes.
DEFAULT_PARITY_DISCLAIMER = (
    "Package and service modes share capability_id, semver policy, and I/O schemas. "
    "They may differ in defaults, available tools/secrets (service-only provider bindings), "
    "latency, streaming fidelity, interrupt/resume mechanics, and observability depth "
    "(package: subgraph traces; service: boundary events only)."
)


@dataclass(frozen=True)
class ParityReport:
    """Result of comparing package and service implementations of one capability."""

    capability_id: str
    package_version: str | None
    service_version: str | None
    level: ParityLevel
    same_input_schema: bool
    same_output_schema: bool
    notes: tuple[str, ...] = ()
    disclaimer: str = DEFAULT_PARITY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "package_version": self.package_version,
            "service_version": self.service_version,
            "level": self.level,
            "same_input_schema": self.same_input_schema,
            "same_output_schema": self.same_output_schema,
            "notes": list(self.notes),
            "disclaimer": self.disclaimer,
        }


def _schema_name(schema: type[Any] | None) -> str | None:
    if schema is None:
        return None
    return getattr(schema, "__name__", str(schema))


def compare_capability_parity(
    package: GraphCapability[Any, Any, Any] | None,
    service: ServiceCapability[Any, Any, Any] | None,
    /,
    *,
    extra_notes: Sequence[str] = (),
) -> ParityReport:
    """Compare package/service handles; schemas are the API, not implementation details."""
    if package is None and service is None:
        raise CapabilityContractError("Need at least one of package or service")

    pkg_spec = package.spec if package else None
    svc_spec = service.spec if service else None
    capability_id = (pkg_spec or svc_spec).capability_id  # type: ignore[union-attr]

    same_in = (
        pkg_spec is not None
        and svc_spec is not None
        and pkg_spec.input_schema is svc_spec.input_schema
    ) or (
        pkg_spec is not None
        and svc_spec is not None
        and _schema_name(pkg_spec.input_schema) == _schema_name(svc_spec.input_schema)
    )
    same_out = (
        pkg_spec is not None
        and svc_spec is not None
        and pkg_spec.output_schema is svc_spec.output_schema
    ) or (
        pkg_spec is not None
        and svc_spec is not None
        and _schema_name(pkg_spec.output_schema) == _schema_name(svc_spec.output_schema)
    )

    notes: list[str] = list(extra_notes)
    if package is None:
        notes.append("No package delivery registered.")
        level: ParityLevel = "unknown"
    elif service is None:
        notes.append("No service delivery registered.")
        level = "unknown"
    elif not same_in or not same_out:
        notes.append("Input/output schema identity differs between package and service.")
        level = "divergent"
    elif pkg_spec and svc_spec and pkg_spec.version == svc_spec.version:
        notes.append("Same semver and aligned I/O schema names; runtime behavior may still differ.")
        level = "identical_io"
    elif pkg_spec and svc_spec and pkg_spec.semver.major == svc_spec.semver.major:
        notes.append(
            f"Major-aligned versions (package {pkg_spec.version}, service {svc_spec.version})."
        )
        level = "compatible_major"
    else:
        notes.append("Versions cross a major boundary; treat as potentially breaking.")
        level = "divergent"

    return ParityReport(
        capability_id=capability_id,
        package_version=pkg_spec.version if pkg_spec else None,
        service_version=svc_spec.version if svc_spec else None,
        level=level,
        same_input_schema=bool(same_in),
        same_output_schema=bool(same_out),
        notes=tuple(notes),
    )


def assert_io_compatible(
    left: CapabilitySpec[Any, Any, Any],
    right: CapabilitySpec[Any, Any, Any],
    /,
    *,
    require_same_major: bool = True,
) -> None:
    """Raise if two specs are not safe to treat as the same capability line."""
    if left.capability_id != right.capability_id:
        raise CapabilityContractError(
            f"capability_id mismatch: {left.capability_id!r} vs {right.capability_id!r}"
        )
    if require_same_major and left.semver.major != right.semver.major:
        raise CapabilityContractError(
            f"Major semver mismatch for {left.capability_id}: "
            f"{left.version} vs {right.version}"
        )
    if _schema_name(left.input_schema) != _schema_name(right.input_schema):
        raise CapabilityContractError(
            f"input_schema mismatch for {left.capability_id}: "
            f"{_schema_name(left.input_schema)} vs {_schema_name(right.input_schema)}"
        )
    if _schema_name(left.output_schema) != _schema_name(right.output_schema):
        raise CapabilityContractError(
            f"output_schema mismatch for {left.capability_id}: "
            f"{_schema_name(left.output_schema)} vs {_schema_name(right.output_schema)}"
        )


def is_breaking_schema_change(old: CapabilitySpec[Any, Any, Any], new: CapabilitySpec[Any, Any, Any]) -> bool:
    """Heuristic: different I/O schema types or major semver bump => breaking."""
    if old.capability_id != new.capability_id:
        return True
    if new.semver.is_breaking_change_from(old.semver):
        return True
    return (
        _schema_name(old.input_schema) != _schema_name(new.input_schema)
        or _schema_name(old.output_schema) != _schema_name(new.output_schema)
    )
