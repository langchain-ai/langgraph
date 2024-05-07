class GraphRecursionError(RecursionError):
    pass


class EmptyChannelError(Exception):
    """Raised when attempting to get the value of a channel that hasn't been updated
    for the first time yet."""

    pass


class InvalidUpdateError(Exception):
    """Raised when attempting to update a channel with an invalid sequence of updates."""

    pass
