"""Multi-version service windows and version policy helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from langgraph.capability.contract import CapabilitySpec, SemVer, select_capability_version
from langgraph.capability.errors import CapabilityVersionError
from langgraph.capability.service import ServiceCapability, ServiceEndpoint


@dataclass(frozen=True)
class VersionWindow:
    """A supported semver (or major line) with optional retirement date."""

    version: str
    """Exact semver (``1.2.3``) or major/minor request form (``1``, ``1.2``)."""

    retired_after: date | None = None
    """After this date (UTC calendar day), the version is outside the support window."""

    label: str = ""
    """Optional deploy label / channel (e.g. ``stable``, ``n-1``)."""

    def is_active(self, *, on: date | None = None) -> bool:
        today = on or datetime.now(timezone.utc).date()
        if self.retired_after is None:
            return True
        return today <= self.retired_after


@dataclass
class ServiceVersionPolicy:
    """Declare which capability versions a service endpoint family supports.

    Typical org policy: pin clients to a **major**; providers keep current + N-1.
    """

    capability_id: str
    windows: list[VersionWindow] = field(default_factory=list)
    prefer_latest_in_major: bool = True

    def active_windows(self, *, on: date | None = None) -> list[VersionWindow]:
        return [w for w in self.windows if w.is_active(on=on)]

    def supports(self, version_request: str, *, on: date | None = None) -> bool:
        try:
            self.resolve(version_request, on=on)
            return True
        except CapabilityVersionError:
            return False

    def resolve(self, version_request: str, *, on: date | None = None) -> VersionWindow:
        """Pick the best active window satisfying the request."""
        active = self.active_windows(on=on)
        if not active:
            raise CapabilityVersionError(
                f"No active version windows for {self.capability_id!r}"
            )

        # Build pseudo-specs from windows that are exact semver for selection,
        # else match request against window.version directly.
        exact_specs: list[CapabilitySpec[Any, Any, Any]] = []
        fuzzy: list[VersionWindow] = []
        for w in active:
            try:
                SemVer.parse(w.version)

                class _In:
                    pass

                class _Out:
                    pass

                exact_specs.append(
                    CapabilitySpec(
                        capability_id=self.capability_id,
                        version=w.version,
                        input_schema=_In,
                        output_schema=_Out,
                    )
                )
            except CapabilityVersionError:
                fuzzy.append(w)

        req = version_request.strip()
        # Direct window hit (major line ``1`` etc.)
        for w in active:
            if w.version == req:
                return w
            try:
                sv = SemVer.parse(w.version)
                if sv.compatible_with_request(req):
                    # collect later for highest
                    pass
            except CapabilityVersionError:
                if req in {"*", "latest"} and w.label in {"stable", "latest", ""}:
                    pass

        if exact_specs:
            try:
                chosen = select_capability_version(
                    exact_specs, self.capability_id, req
                )
                for w in active:
                    if w.version == chosen.version:
                        return w
            except CapabilityVersionError:
                pass

        # Fuzzy: window stores ``1`` or ``1.2``
        matching_fuzzy = []
        for w in fuzzy:
            try:
                # Request exact semver against major/minor window
                if _SEMVER_FULL.match(req):
                    sv = SemVer.parse(req)
                    if sv.compatible_with_request(w.version) or w.version == str(
                        sv.major
                    ):
                        matching_fuzzy.append(w)
                elif w.version == req or req in {"*", "latest"}:
                    matching_fuzzy.append(w)
            except CapabilityVersionError:
                if w.version == req:
                    matching_fuzzy.append(w)

        if matching_fuzzy:
            return matching_fuzzy[-1] if self.prefer_latest_in_major else matching_fuzzy[0]

        # Last pass: any active exact semver compatible with request
        for w in active:
            try:
                sv = SemVer.parse(w.version)
                if sv.compatible_with_request(req):
                    return w
            except CapabilityVersionError:
                continue

        raise CapabilityVersionError(
            f"Version {version_request!r} not in active window for "
            f"{self.capability_id!r}; active={[w.version for w in active]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "prefer_latest_in_major": self.prefer_latest_in_major,
            "windows": [
                {
                    "version": w.version,
                    "retired_after": w.retired_after.isoformat()
                    if w.retired_after
                    else None,
                    "label": w.label,
                    "active": w.is_active(),
                }
                for w in self.windows
            ],
        }


import re

_SEMVER_FULL = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def bind_service_to_policy(
    capability: ServiceCapability[Any, Any, Any],
    policy: ServiceVersionPolicy,
    version_request: str = "*",
    /,
    *,
    on: date | None = None,
) -> ServiceCapability[Any, Any, Any]:
    """Return a shallow copy of the service capability pinned to a policy window."""
    if policy.capability_id != capability.capability_id:
        raise CapabilityVersionError(
            f"Policy capability_id {policy.capability_id!r} does not match "
            f"{capability.capability_id!r}"
        )
    window = policy.resolve(version_request, on=on)
    ep = capability.endpoint
    new_ep = ServiceEndpoint(
        url=ep.url,
        assistant_id=ep.assistant_id,
        api_key=ep.api_key,
        headers=ep.headers,
        graph_id=ep.graph_id,
        version_label=window.version,
    )
    return ServiceCapability(
        spec=capability.spec,
        endpoint=new_ep,
        validate_io=capability.validate_io,
        invoker=capability.invoker,
        async_invoker=capability.async_invoker,
        remote_graph=capability.remote_graph,
    )


def default_n_minus_one_policy(
    capability_id: str,
    current: str,
    previous: str | None = None,
    /,
    *,
    previous_retired_after: date | None = None,
) -> ServiceVersionPolicy:
    """Helper: support ``current`` fully and optional ``previous`` major/minor until a date."""
    windows = [VersionWindow(version=current, label="current")]
    if previous:
        windows.append(
            VersionWindow(
                version=previous,
                retired_after=previous_retired_after,
                label="n-1",
            )
        )
    return ServiceVersionPolicy(capability_id=capability_id, windows=windows)
