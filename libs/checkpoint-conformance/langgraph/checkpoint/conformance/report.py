"""Capability report: results, progress callbacks, and pretty-printing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.conformance.capabilities import (
    BASE_CAPABILITIES,
    EXTENDED_CAPABILITIES,
    Capability,
)

# Callback type for per-test progress reporting.
# (capability_name, test_name, passed, error_msg_or_None) -> None
OnTestResult = Callable[[str, str, bool, str | None], None]

# Callback type for capability-level events.
# (capability_name, detected) -> None
OnCapabilityStart = Callable[[str, bool], None]


class ProgressCallbacks:
    """Grouped callbacks for progress reporting during validation."""

    def __init__(
        self,
        *,
        on_capability_start: Callable[[str, bool], None] | None = None,
        on_test_result: OnTestResult | None = None,
        on_capability_end: Callable[[str], None] | None = None,
    ) -> None:
        self.on_capability_start = on_capability_start
        self.on_test_result = on_test_result
        self.on_capability_end = on_capability_end

    @classmethod
    def default(cls) -> ProgressCallbacks:
        """Dot-style progress: ``.`` per pass, ``F`` per fail."""

        def _cap_start(capability: str, detected: bool) -> None:
            if detected:
                print(f"  {capability}: ", end="", flush=True)
            else:
                print(f"  ⊘ {capability} (not implemented)")

        def _test_result(
            capability: str, test_name: str, passed: bool, error: str | None
        ) -> None:
            print("." if passed else "F", end="", flush=True)

        def _cap_end(capability: str) -> None:
            print()  # newline after dots

        return cls(
            on_capability_start=_cap_start,
            on_test_result=_test_result,
            on_capability_end=_cap_end,
        )

    @classmethod
    def verbose(cls) -> ProgressCallbacks:
        """Per-test output with names and errors."""

        def _cap_start(capability: str, detected: bool) -> None:
            if detected:
                print(f"  {capability}:")
            else:
                print(f"  ⊘ {capability} (not implemented)")

        def _test_result(
            capability: str, test_name: str, passed: bool, error: str | None
        ) -> None:
            icon = "✓" if passed else "✗"
            print(f"    {icon} {test_name}")
            if error:
                for line in error.rstrip().splitlines():
                    print(f"      {line}")

        return cls(
            on_capability_start=_cap_start,
            on_test_result=_test_result,
        )

    @classmethod
    def quiet(cls) -> ProgressCallbacks:
        """No progress output."""
        return cls()


@dataclass
class CapabilityResult:
    """Result of running a single capability's test suite."""

    detected: bool = False
    passed: bool | None = None  # None = skipped
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    failures: list[str] = field(default_factory=list)


@dataclass
class CapabilityReport:
    """Aggregate report across all capabilities."""

    checkpointer_name: str
    results: dict[str, CapabilityResult] = field(default_factory=dict)

    def passed_all_base(self) -> bool:
        """Whether all base capability tests passed."""
        for cap in BASE_CAPABILITIES:
            result = self.results.get(cap.value)
            if result is None or result.passed is not True:
                return False
        return True

    def passed_all(self) -> bool:
        """Whether every detected capability's tests passed."""
        for result in self.results.values():
            if result.detected and result.passed is not True:
                return False
        return True

    def conformance_level(self) -> str:
        """Return a human-readable conformance level string."""
        if self.passed_all():
            return "FULL"
        if self.passed_all_base():
            return "BASE+PARTIAL"
        return "BASE" if self._any_base_passed() else "NONE"

    def _any_base_passed(self) -> bool:
        for cap in BASE_CAPABILITIES:
            result = self.results.get(cap.value)
            if result and result.passed is True:
                return True
        return False

    def print_report(self) -> None:
        """Pretty-print the report to stdout."""
        width = 52
        border = "=" * width
        print(f"\n{'':>2}{border}")
        print(f"{'':>2}  Checkpointer Validation: {self.checkpointer_name}")
        print(f"{'':>2}{border}")

        def _section(title: str, caps: frozenset[Capability]) -> None:
            print(f"{'':>2}  {title}")
            for cap in sorted(caps, key=lambda c: c.value):
                result = self.results.get(cap.value)
                if result is None:
                    icon = "  "
                    suffix = "(no tests)"
                elif not result.detected:
                    icon = "⊘ "
                    suffix = "(not implemented)"
                elif result.passed is True:
                    icon = "✅"
                    suffix = ""
                elif result.passed is False:
                    icon = "❌"
                    suffix = f"({result.tests_failed} failed)"
                else:
                    icon = "⏭ "
                    suffix = "(skipped)"
                print(f"{'':>2}    {icon} {cap.value:20s} {suffix}")
            print()

        _section("BASE CAPABILITIES", BASE_CAPABILITIES)
        _section("EXTENDED CAPABILITIES", EXTENDED_CAPABILITIES)

        total = sum(1 for r in self.results.values() if r.detected)
        passed = sum(
            1 for r in self.results.values() if r.detected and r.passed is True
        )
        level = self.conformance_level()
        print(f"{'':>2}  Result: {level} ({passed}/{total})")
        print(f"{'':>2}{border}\n")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "checkpointer_name": self.checkpointer_name,
            "conformance_level": self.conformance_level(),
            "results": {
                name: {
                    "detected": r.detected,
                    "passed": r.passed,
                    "tests_passed": r.tests_passed,
                    "tests_failed": r.tests_failed,
                    "tests_skipped": r.tests_skipped,
                    "failures": r.failures,
                }
                for name, r in self.results.items()
            },
        }
