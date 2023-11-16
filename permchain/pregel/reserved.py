from enum import StrEnum


class ReservedChannels(StrEnum):
    """Channels managed by the framework."""

    is_last_step = "is_last_step"
    """A channel that is True if the current step is the last step, False otherwise."""

    idempotency_key = "idempotency_key"
    """A channel that contains a stable string id for the current step.
    Can be used for caching or ensuring idempotency of side effects."""
