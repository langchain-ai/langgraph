from typing import Any

CONFIG_KEY_SEND = "__pregel_send"
CONFIG_KEY_READ = "__pregel_read"

INTERRUPT = "__interrupt__"
TASKS = "__pregel_tasks"

RESERVED = {INTERRUPT, TASKS, CONFIG_KEY_SEND, CONFIG_KEY_READ}

TAG_HIDDEN = "langsmith:hidden"


class Send:
    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )
