from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Config(_message.Message):
    __slots__ = ("checkpoint_ns",)
    CHECKPOINT_NS_FIELD_NUMBER: _ClassVar[int]
    checkpoint_ns: str
    def __init__(self, checkpoint_ns: _Optional[str] = ...) -> None: ...

class PregelExecutableTask(_message.Message):
    __slots__ = ("task_id", "name", "input", "config", "path")
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    name: str
    input: _containers.RepeatedScalarFieldContainer[str]
    config: Config
    path: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, task_id: _Optional[str] = ..., name: _Optional[str] = ..., input: _Optional[_Iterable[str]] = ..., config: _Optional[_Union[Config, _Mapping]] = ..., path: _Optional[_Iterable[str]] = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("write", "error")
    class Write(_message.Message):
        __slots__ = ("name", "value")
        NAME_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        name: str
        value: bytes
        def __init__(self, name: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    class Error(_message.Message):
        __slots__ = ("name", "value")
        NAME_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        name: str
        value: bytes
        def __init__(self, name: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    WRITE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    write: Event.Write
    error: Event.Error
    def __init__(self, write: _Optional[_Union[Event.Write, _Mapping]] = ..., error: _Optional[_Union[Event.Error, _Mapping]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListGraphsResponse(_message.Message):
    __slots__ = ("graphs",)
    class Graph(_message.Message):
        __slots__ = ("nodes", "channel_names")
        class Node(_message.Message):
            __slots__ = ("name", "input")
            NAME_FIELD_NUMBER: _ClassVar[int]
            INPUT_FIELD_NUMBER: _ClassVar[int]
            name: str
            input: _containers.RepeatedScalarFieldContainer[str]
            def __init__(self, name: _Optional[str] = ..., input: _Optional[_Iterable[str]] = ...) -> None: ...
        NODES_FIELD_NUMBER: _ClassVar[int]
        CHANNEL_NAMES_FIELD_NUMBER: _ClassVar[int]
        nodes: _containers.RepeatedCompositeFieldContainer[ListGraphsResponse.Graph.Node]
        channel_names: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, nodes: _Optional[_Iterable[_Union[ListGraphsResponse.Graph.Node, _Mapping]]] = ..., channel_names: _Optional[_Iterable[str]] = ...) -> None: ...
    GRAPHS_FIELD_NUMBER: _ClassVar[int]
    graphs: _containers.RepeatedCompositeFieldContainer[ListGraphsResponse.Graph]
    def __init__(self, graphs: _Optional[_Iterable[_Union[ListGraphsResponse.Graph, _Mapping]]] = ...) -> None: ...
