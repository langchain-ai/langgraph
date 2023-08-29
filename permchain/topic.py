from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    TypedDict,
)

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema.runnable import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain.schema.runnable.base import Other, coerce_to_runnable

from permchain.constants import CONFIG_GET_KEY, CONFIG_SEND_KEY

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


INPUT_TOPIC = "__in__"
OUTPUT_TOPIC = "__out__"
INFLIGHT_TOPIC = "__inflight__"


class InflightMessage(TypedDict):
    message: Any
    topic_name: str
    started_at: str


class Topic(Serializable, Generic[T], ABC):
    name: str

    def __init__(self, name: str):
        super().__init__(name=name)

    def subscribe(self) -> RunnableSubscriber[T]:
        return RunnableSubscriber(topic=self)

    def current(self) -> RunnableCurrentValue[T]:
        return RunnableCurrentValue(topic=self)

    def publish(self) -> RunnablePublisher[T]:
        return RunnablePublisher(topic=self)

    def publish_each(self) -> Runnable[T, T]:
        return RunnablePublisherEach(topic=self)

    @classmethod
    @property
    def IN(cls) -> Topic[T_in]:
        return cls[T_in](INPUT_TOPIC)

    @classmethod
    @property
    def OUT(cls) -> Topic[T_out]:
        return cls[T_out](OUTPUT_TOPIC)

    @classmethod
    @property
    def INFLIGHT(cls) -> Topic[InflightMessage]:
        return cls[InflightMessage](INFLIGHT_TOPIC)


class RunnableConfigForPubSub(RunnableConfig):
    send: Callable[[str, Any], None]
    get: Callable[[str], Any]


class RunnableSubscriber(RunnableBinding[T, Any]):
    topic: Topic[T]

    bound: Runnable[T, Any] = Field(default_factory=RunnablePassthrough)

    kwargs: Mapping[str, Any] = Field(default_factory=dict)

    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other]],
    ) -> RunnableSequence[T, Other]:
        if isinstance(self.bound, RunnablePassthrough):
            return RunnableSubscriber(topic=self.topic, bound=coerce_to_runnable(other))
        else:
            return RunnableSubscriber(topic=self.topic, bound=self.bound | other)

    def __ror__(
        self,
        other: Runnable[Other, Any]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Other, Any] | Callable[[Other], Any]],
    ) -> RunnableSequence[Other, Any]:
        raise NotImplementedError()


class RunnablePublisher(Serializable, Runnable[T, T]):
    topic: Topic[T]

    def invoke(self, input: T, config: Optional[RunnableConfigForPubSub] = None) -> T:
        send = config.get(CONFIG_SEND_KEY, None)
        if send is not None:
            send(self.topic.name, input)
        return input


class RunnablePublisherEach(RunnablePublisher[Sequence[T]]):
    topic: Topic[T]

    def invoke(
        self, input: Sequence[T], config: Optional[RunnableConfigForPubSub] = None
    ) -> Sequence[T]:
        for item in input:
            super().invoke(item, config)


class RunnableCurrentValue(Serializable, Runnable[Any, T]):
    topic: Topic[T]

    def invoke(self, input: T, config: Optional[RunnableConfigForPubSub] = None) -> T:
        get: Callable[[str], None] = config.get(CONFIG_GET_KEY, None)
        if get is not None:
            return get(self.topic.name)
        else:
            raise ValueError("Cannot get value in this context")
