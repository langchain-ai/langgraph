from langgraph.channels.base import BaseChannel as BaseChannel
from langgraph.pregel.read import PregelNode
from langgraph.types import All as All
from typing import Any, Mapping, Sequence

def validate_graph(nodes: Mapping[str, PregelNode], channels: dict[str, BaseChannel], input_channels: str | Sequence[str], output_channels: str | Sequence[str], stream_channels: str | Sequence[str] | None, interrupt_after_nodes: All | Sequence[str], interrupt_before_nodes: All | Sequence[str]) -> None: ...
def validate_keys(keys: str | Sequence[str] | None, channels: Mapping[str, Any]) -> None: ...
