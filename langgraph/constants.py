from typing import Any

from langgraph.errors import InvalidUpdateError

CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"

INTERRUPT = "__interrupt__"
TASKS = "__pregel_tasks"

RESERVED = {INTERRUPT, TASKS, CONFIG_KEY_SEND, CONFIG_KEY_READ}

TAG_HIDDEN = "langsmith:hidden"


class Packet:
    def __init__(self, /, __node__: str, **kwargs: Any) -> None:
        if not kwargs:
            raise InvalidUpdateError(
                "Packet must have at least one keyword argument to pass to node"
            )
        self.node = __node__
        self.kwargs = kwargs
