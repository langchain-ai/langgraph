from _typeshed import Incomplete
from langchain_core.runnables import Runnable, RunnableConfig as RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph.constants import Send
from langgraph.utils.runnable import RunnableCallable
from typing import Any, Callable, NamedTuple, Sequence, TypeVar

TYPE_SEND = Callable[[Sequence[tuple[str, Any]]], None]
R = TypeVar('R', bound=Runnable)
SKIP_WRITE: Incomplete
PASSTHROUGH: Incomplete

class ChannelWriteEntry(NamedTuple):
    channel: str
    value: Any = ...
    skip_none: bool = ...
    mapper: Callable | None = ...

class ChannelWriteTupleEntry(NamedTuple):
    mapper: Callable[[Any], Sequence[tuple[str, Any]] | None]
    value: Any = ...

class ChannelWrite(RunnableCallable):
    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]
    require_at_least_one_of: Sequence[str] | None
    def __init__(self, writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send], *, tags: Sequence[str] | None = None, require_at_least_one_of: Sequence[str] | None = None) -> None: ...
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str: ...
    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]: ...
    @staticmethod
    def do_write(config: RunnableConfig, writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send], require_at_least_one_of: Sequence[str] | None = None) -> None: ...
    @staticmethod
    def is_writer(runnable: Runnable) -> bool: ...
    @staticmethod
    def register_writer(runnable: R) -> R: ...
