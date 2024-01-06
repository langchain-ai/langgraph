from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict
from langchain_core.pydantic_v1 import dataclasses

from permchain.checkpoint.base import BaseCheckpointAdapter
from permchain.pregel import Pregel

dataclasses.DataclassClassOrWrapper

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from permchain.pregel.read import ChannelInvoke


def on_change(field_name: str):
    def decorator(func):
        return ChannelInvoke(bound=RunnableLambda(func), triggers=[field_name])

    return decorator


class DataclassProtocol:
    __dataclass_fields__: ClassVar[Dict[str, Any]]


def thread(data: type[DataclassProtocol], thread_id: str, saver: BaseCheckpointAdapter):
    return Pregel(
        chains={
            v.__name__: v
            for v in data.__dict__.values()
            if isinstance(v, ChannelInvoke)
        }
    )


@dataclass
class Agent:
    messages: list[BaseMessage] = field(default_factory=list)
    actions: list[BaseMessage] = field(default_factory=list)

    @on_change("messages")
    def plan(self):
        ...

    @on_change("actions")
    def execute(self):
        ...
