"""Core conformance runner — detects capabilities, runs test suites, builds report."""

from __future__ import annotations

from langgraph.checkpoint.conformance.capabilities import (
    Capability,
    DetectedCapabilities,
)
from langgraph.checkpoint.conformance.initializer import RegisteredCheckpointer
from langgraph.checkpoint.conformance.report import (
    CapabilityReport,
    CapabilityResult,
    ProgressCallbacks,
)
from langgraph.checkpoint.conformance.spec.test_copy_thread import run_copy_thread_tests
from langgraph.checkpoint.conformance.spec.test_delete_for_runs import (
    run_delete_for_runs_tests,
)
from langgraph.checkpoint.conformance.spec.test_delete_thread import (
    run_delete_thread_tests,
)
from langgraph.checkpoint.conformance.spec.test_get_tuple import run_get_tuple_tests
from langgraph.checkpoint.conformance.spec.test_list import run_list_tests
from langgraph.checkpoint.conformance.spec.test_prune import run_prune_tests
from langgraph.checkpoint.conformance.spec.test_put import run_put_tests
from langgraph.checkpoint.conformance.spec.test_put_writes import run_put_writes_tests

# Maps capability to its runner function.
_RUNNERS = {
    Capability.PUT: run_put_tests,
    Capability.PUT_WRITES: run_put_writes_tests,
    Capability.GET_TUPLE: run_get_tuple_tests,
    Capability.LIST: run_list_tests,
    Capability.DELETE_THREAD: run_delete_thread_tests,
    Capability.DELETE_FOR_RUNS: run_delete_for_runs_tests,
    Capability.COPY_THREAD: run_copy_thread_tests,
    Capability.PRUNE: run_prune_tests,
}


async def validate(
    registered: RegisteredCheckpointer,
    *,
    capabilities: set[str] | None = None,
    progress: ProgressCallbacks | None = None,
) -> CapabilityReport:
    """Run the validation suite against a registered checkpointer.

    Args:
        registered: A RegisteredCheckpointer (from @checkpointer_test decorator).
        capabilities: If given, only run tests for these capability names.
            Otherwise, auto-detect and run all applicable tests.
        progress: Optional progress callbacks for incremental output.
            Use ``ProgressCallbacks.default()`` for dot-style,
            ``ProgressCallbacks.verbose()`` for per-test output, or
            ``None`` / ``ProgressCallbacks.quiet()`` for silent mode.

    Returns:
        A CapabilityReport with per-capability results.
    """
    report = CapabilityReport(checkpointer_name=registered.name)

    # Determine which capabilities to test.
    caps_to_test: set[Capability]
    if capabilities is not None:
        caps_to_test = {Capability(c) for c in capabilities}
    else:
        caps_to_test = set(Capability)

    async with registered.enter_lifespan():
        for cap in Capability:
            if cap in caps_to_test and cap.value not in registered.skip_capabilities:
                # Create a fresh checkpointer for each capability suite.
                async with registered.create() as saver:
                    detected = DetectedCapabilities.from_instance(saver)
                    is_detected = cap in detected.detected

                    if not is_detected:
                        if progress and progress.on_capability_start:
                            progress.on_capability_start(cap.value, False)
                        report.results[cap.value] = CapabilityResult(
                            detected=False,
                            passed=None,
                            tests_skipped=1,
                        )
                        continue

                    runner = _RUNNERS.get(cap)
                    if runner is None:
                        report.results[cap.value] = CapabilityResult(
                            detected=True,
                            passed=None,
                            tests_skipped=1,
                        )
                        continue

                    if progress and progress.on_capability_start:
                        progress.on_capability_start(cap.value, True)

                    passed, failed, failures = await runner(
                        saver,
                        on_test_result=progress.on_test_result if progress else None,
                    )

                    if progress and progress.on_capability_end:
                        progress.on_capability_end(cap.value)

                    report.results[cap.value] = CapabilityResult(
                        detected=True,
                        passed=failed == 0,
                        tests_passed=passed,
                        tests_failed=failed,
                        failures=failures,
                    )
            else:
                if progress and progress.on_capability_start:
                    progress.on_capability_start(cap.value, False)
                report.results[cap.value] = CapabilityResult(
                    detected=False,
                    passed=None,
                    tests_skipped=1,
                )

    return report
