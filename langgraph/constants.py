from typing import Any, NamedTuple

CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"

INTERRUPT = "__interrupt__"
TASKS = "__pregel_tasks"

RESERVED = {INTERRUPT, TASKS, CONFIG_KEY_SEND, CONFIG_KEY_READ}

TAG_HIDDEN = "langsmith:hidden"


class Packet(NamedTuple):
    node: str
    arg: Any
