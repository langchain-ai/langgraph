"""Capability contracts: stable IDs, semver, I/O schemas, side-effect declarations."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from langgraph.capability.errors import CapabilityContractError, CapabilityVersionError

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ParamsT = TypeVar("ParamsT")

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)


class SideEffect(str, Enum):
    """Declared external effects a capability may perform."""

    NONE = "none"
    READ_EXTERNAL = "read_external"
    WRITE_EXTERNAL = "write_external"
    TOOLS = "tools"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    NETWORK = "network"
    MODEL_CALLS = "model_calls"


class StateBoundary(str, Enum):
    """How capability state relates to a parent graph."""

    ISOLATED = "isolated"
    """Child owns private state; parent only sees input/output at the boundary."""

    MAPPED = "mapped"
    """Parent maps selected fields to/from capability input/output schemas."""

    SHARED = "shared"
    """Parent and child share state channels (disallowed for published library capabilities)."""


@dataclass(frozen=True, slots=True)
class SemVer:
    """Parsed semantic version for capability implementations."""

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None
    raw: str = ""

    @classmethod
    def parse(cls, version: str) -> SemVer:
        match = _SEMVER_RE.match(version.strip())
        if not match:
            raise CapabilityVersionError(
                f"Invalid semver {version!r}; expected MAJOR.MINOR.PATCH"
            )
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("build"),
            raw=version.strip(),
        )

    def __str__(self) -> str:
        return self.raw or f"{self.major}.{self.minor}.{self.patch}"

    def compatible_with_request(self, requested: str) -> bool:
        """Return True if this version satisfies a request like ``1``, ``1.2``, or ``1.2.3``."""
        requested = requested.strip()
        if requested in {"*", "latest"}:
            return self.prerelease is None
        if _SEMVER_RE.match(requested):
            other = SemVer.parse(requested)
            return (
                self.major == other.major
                and self.minor == other.minor
                and self.patch == other.patch
                and (self.prerelease or "") == (other.prerelease or "")
            )
        parts = requested.split(".")
        try:
            nums = [int(p) for p in parts]
        except ValueError as exc:
            raise CapabilityVersionError(
                f"Unsupported version request {requested!r}"
            ) from exc
        if len(nums) == 1:
            return self.major == nums[0] and self.prerelease is None
        if len(nums) == 2:
            return (
                self.major == nums[0]
                and self.minor == nums[1]
                and self.prerelease is None
            )
        raise CapabilityVersionError(f"Unsupported version request {requested!r}")

    def is_breaking_change_from(self, other: SemVer) -> bool:
        return self.major != other.major


@dataclass(frozen=True)
class CapabilitySpec(Generic[InputT, OutputT, ParamsT]):
    """Contract for a reusable graph capability (package and/or service).

    Consumers depend on ``capability_id``, ``version``, and schemas—not internal
    node names or private state channels.
    """

    capability_id: str
    version: str
    input_schema: type[InputT]
    output_schema: type[OutputT]
    public_params_schema: type[ParamsT] | None = None
    side_effects: frozenset[SideEffect] = field(default_factory=frozenset)
    state_boundary: StateBoundary = StateBoundary.ISOLATED
    description: str = ""
    owner: str = ""
    documentation_url: str = ""
    error_model: str = (
        "Boundary failures raise CapabilityInvocationError (or subclasses) with "
        "capability_id/version/run_id when available. Schema violations raise "
        "CapabilitySchemaError. Unsupported versions raise CapabilityVersionError."
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    semver: SemVer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.capability_id or not self.capability_id.strip():
            raise CapabilityContractError("capability_id is required")
        if "/" in self.capability_id or " " in self.capability_id:
            raise CapabilityContractError(
                "capability_id must be a stable dotted/colon identifier without spaces "
                f"(got {self.capability_id!r})"
            )
        object.__setattr__(self, "semver", SemVer.parse(self.version))
        if self.state_boundary is StateBoundary.SHARED:
            raise CapabilityContractError(
                "SHARED state_boundary is disallowed for published capabilities; "
                "use ISOLATED or MAPPED so parents only depend on I/O schemas."
            )
        if self.input_schema is None or self.output_schema is None:
            raise CapabilityContractError("input_schema and output_schema are required")

    def supports_version_request(self, requested: str) -> bool:
        return self.semver.compatible_with_request(requested)

    def to_metadata(self) -> dict[str, Any]:
        """Serializable metadata for catalogs, deploy tags, and observability."""
        return {
            "capability_id": self.capability_id,
            "version": self.version,
            "input_schema": getattr(self.input_schema, "__name__", str(self.input_schema)),
            "output_schema": getattr(
                self.output_schema, "__name__", str(self.output_schema)
            ),
            "public_params_schema": (
                getattr(self.public_params_schema, "__name__", str(self.public_params_schema))
                if self.public_params_schema is not None
                else None
            ),
            "side_effects": sorted(se.value for se in self.side_effects),
            "state_boundary": self.state_boundary.value,
            "description": self.description,
            "owner": self.owner,
            "documentation_url": self.documentation_url,
            "error_model": self.error_model,
            **dict(self.metadata),
        }


Builder = Callable[..., Any]
"""Callable that builds a graph given public params (package entrypoint)."""


def validate_capability_spec(spec: CapabilitySpec[Any, Any, Any]) -> CapabilitySpec[Any, Any, Any]:
    """Re-run contract checks; returns the same spec if valid."""
    # Construction already validates; this is for external/dynamic specs.
    if not isinstance(spec, CapabilitySpec):
        raise CapabilityContractError("spec must be a CapabilitySpec instance")
    SemVer.parse(spec.version)
    return spec


def select_capability_version(
    available: Sequence[CapabilitySpec[Any, Any, Any]],
    capability_id: str,
    version_request: str = "*",
) -> CapabilitySpec[Any, Any, Any]:
    """Pick the highest semver implementation matching id + version request."""
    candidates = [s for s in available if s.capability_id == capability_id]
    if not candidates:
        raise CapabilityVersionError(
            f"No capability registered with id {capability_id!r}"
        )
    matching = [s for s in candidates if s.supports_version_request(version_request)]
    if not matching:
        versions = ", ".join(sorted({s.version for s in candidates}))
        raise CapabilityVersionError(
            f"No version of {capability_id!r} satisfies {version_request!r}; "
            f"available: {versions}"
        )

    def sort_key(spec: CapabilitySpec[Any, Any, Any]) -> tuple[int, int, int, int]:
        v = spec.semver
        # Prefer non-prerelease; then highest major.minor.patch
        pre = 0 if v.prerelease is None else 1
        return (pre, -v.major, -v.minor, -v.patch)

    return sorted(matching, key=sort_key)[0]
