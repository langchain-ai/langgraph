from langgraph.utils import StrEnum


class ReservedChannels(StrEnum):
    """Channels managed by the framework."""

    is_last_step = "is_last_step"
    """A channel that is True if the current step is the last step, False otherwise."""


AllReservedChannels = {channel.value for channel in ReservedChannels}
