"""CCS Runtime Governance Layer for LangGraph.

Provides formal behavioral conformance verification based on CCS 1.0 specification.
This module wraps any BaseCheckpointSaver with runtime state sovereignty validation.

References:
    - CCS 1.0 Specification: https://github.com/Correctover/ccs-integration-kit/releases/tag/v1.0.0
    - DOI: 10.5281/zenodo.21234580
"""

from ccs_verifier.checkpointer import (
    CCSVerifierCheckpointer,
    CCSConformanceError,
    Required,
    DecisionLog,
    DecisionRecord,
    DecisionOutcome,
)

__version__ = "1.0.0"
__all__ = [
    "CCSVerifierCheckpointer",
    "CCSConformanceError",
    "Required",
    "DecisionLog",
    "DecisionRecord",
    "DecisionOutcome",
]
