import operator
import sys
import time
from collections import defaultdict
from typing import Annotated, Literal

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from typing_extensions import TypedDict

from langgraph._internal._config import patch_configurable
from langgraph.graph.state import StateGraph
from langgraph.pregel._checkpoint import copy_checkpoint
from langgraph.types import Command, Interrupt, PregelTask, StateSnapshot, interrupt
from tests.any_int import AnyInt
from tests.any_str import AnyDict, AnyObject, AnyStr

pytestmark = pytest.mark.anyio

NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)


def get_expected_history(*, exc_task_results: int = 0) -> list[StateSnapshot]:
    return [
        StateSnapshot(
            values={
                "query": "analyzed: query: what is weather in sf",
                "answer": "doc1,doc2,doc3,doc4",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 4,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(),
            interrupts=(),
        ),
        StateSnapshot(
            values={
                "query": "analyzed: query: what is weather in sf",
                "docs": ["doc1", "doc2", "doc3", "doc4"],
            },
            next=("qa",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 3,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="qa",
                    path=("__pregel_pull", "qa"),
                    error=None,
                    interrupts=()
                    if exc_task_results
                    else (
                        Interrupt(
                            value="",
                            id=AnyStr(),
                        ),
                    ),
                    state=None,
                    result=None
                    if exc_task_results
                    else {"answer": "doc1,doc2,doc3,doc4"},
                ),
            ),
            interrupts=()
            if exc_task_results
            else (
                Interrupt(
                    value="",
                    id=AnyStr(),
                ),
            ),
        ),
        StateSnapshot(
            values={
                "query": "analyzed: query: what is weather in sf",
                "docs": ["doc3", "doc4"],
            },
            next=("retriever_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 2,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="retriever_one",
                    path=("__pregel_pull", "retriever_one"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None if exc_task_results else {"docs": ["doc1", "doc2"]},
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values={"query": "query: what is weather in sf", "docs": []},
            next=("analyzer_one", "retriever_two"),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="analyzer_one",
                    path=("__pregel_pull", "analyzer_one"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None
                    if exc_task_results
                    else {"query": "analyzed: query: what is weather in sf"},
                ),
                PregelTask(
                    id=AnyStr(),
                    name="retriever_two",
                    path=("__pregel_pull", "retriever_two"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None
                    if exc_task_results >= 2
                    else {"docs": ["doc3", "doc4"]},
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values={"query": "what is weather in sf", "docs": []},
            next=("rewrite_query",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="rewrite_query",
                    path=("__pregel_pull", "rewrite_query"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result=None
                    if exc_task_results
                    else {"query": "query: what is weather in sf"},
                ),
            ),
            interrupts=(),
        ),
        StateSnapshot(
            values={"docs": []},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "step": -1,
                "parents": {},
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=("__pregel_pull", "__start__"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result={"query": "what is weather in sf"},
                ),
            ),
            interrupts=(),
        ),
    ]


SAVED_CHECKPOINTS = {
    "3": [
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2149-6faa-8004-9d848038f10a",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.237381+00:00",
                "id": "1f00fd5f-2149-6faa-8004-9d848038f10a",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.6697367414225304",
                    "query": "00000000000000000000000000000004.0.18727156933289513",
                    "branch:to:rewrite_query": "00000000000000000000000000000003.0.14126716107927562",
                    "branch:to:analyzer_one": "00000000000000000000000000000004.0.15766851053750708",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.04821745244115927",
                    "branch:to:retriever_one": "00000000000000000000000000000005.0.7710812646219019",
                    "docs": "00000000000000000000000000000005.0.7916507770116351",
                    "branch:to:qa": "00000000000000000000000000000006.0.6375257096095945",
                    "answer": "00000000000000000000000000000006.0.9100669543952636",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.7234984738744598"
                    },
                    "rewrite_query": {
                        "branch:to:rewrite_query": "00000000000000000000000000000002.0.05597832024496252"
                    },
                    "analyzer_one": {
                        "branch:to:analyzer_one": "00000000000000000000000000000003.0.7165779439892241"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.7762711252277583"
                    },
                    "retriever_one": {
                        "branch:to:retriever_one": "00000000000000000000000000000004.0.5907938097782264"
                    },
                    "__interrupt__": {
                        "query": "00000000000000000000000000000004.0.18727156933289513",
                        "docs": "00000000000000000000000000000005.0.7916507770116351",
                        "__start__": "00000000000000000000000000000002.0.6697367414225304",
                        "branch:to:rewrite_query": "00000000000000000000000000000003.0.14126716107927562",
                        "branch:to:analyzer_one": "00000000000000000000000000000004.0.15766851053750708",
                        "branch:to:retriever_one": "00000000000000000000000000000005.0.7710812646219019",
                        "branch:to:retriever_two": "00000000000000000000000000000004.0.04821745244115927",
                        "branch:to:qa": "00000000000000000000000000000005.0.5602643794940962",
                    },
                    "qa": {
                        "branch:to:qa": "00000000000000000000000000000005.0.5602643794940962"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "answer": "doc1,doc2,doc3,doc4",
                },
                "updated_channels": None,
            },
            metadata={
                "source": "loop",
                "step": 4,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2140-6fd6-8003-2051ce36b79c",
                }
            },
            pending_writes=[],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2140-6fd6-8003-2051ce36b79c",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.233695+00:00",
                "id": "1f00fd5f-2140-6fd6-8003-2051ce36b79c",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.6697367414225304",
                    "query": "00000000000000000000000000000004.0.18727156933289513",
                    "branch:to:rewrite_query": "00000000000000000000000000000003.0.14126716107927562",
                    "branch:to:analyzer_one": "00000000000000000000000000000004.0.15766851053750708",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.04821745244115927",
                    "branch:to:retriever_one": "00000000000000000000000000000005.0.7710812646219019",
                    "docs": "00000000000000000000000000000005.0.7916507770116351",
                    "branch:to:qa": "00000000000000000000000000000005.0.5602643794940962",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.7234984738744598"
                    },
                    "rewrite_query": {
                        "branch:to:rewrite_query": "00000000000000000000000000000002.0.05597832024496252"
                    },
                    "analyzer_one": {
                        "branch:to:analyzer_one": "00000000000000000000000000000003.0.7165779439892241"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.7762711252277583"
                    },
                    "retriever_one": {
                        "branch:to:retriever_one": "00000000000000000000000000000004.0.5907938097782264"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "branch:to:qa": None,
                },
                "updated_channels": None,
            },
            metadata={
                "source": "loop",
                "step": 3,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-213c-6940-8002-a28f475a6478",
                }
            },
            pending_writes=[
                (
                    "2430f303-da9f-2e3e-738c-2e8ea28e8973",
                    "__interrupt__",
                    [
                        Interrupt(
                            value="",
                            resumable=True,  # type: ignore[arg-type]
                            ns=["qa:2430f303-da9f-2e3e-738c-2e8ea28e8973"],  # type: ignore[arg-type]
                        )
                    ],
                ),
                ("00000000-0000-0000-0000-000000000000", "__resume__", ""),
                ("2430f303-da9f-2e3e-738c-2e8ea28e8973", "__resume__", [""]),
                (
                    "2430f303-da9f-2e3e-738c-2e8ea28e8973",
                    "answer",
                    "doc1,doc2,doc3,doc4",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-213c-6940-8002-a28f475a6478",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.231890+00:00",
                "id": "1f00fd5f-213c-6940-8002-a28f475a6478",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.6697367414225304",
                    "query": "00000000000000000000000000000004.0.18727156933289513",
                    "branch:to:rewrite_query": "00000000000000000000000000000003.0.14126716107927562",
                    "branch:to:analyzer_one": "00000000000000000000000000000004.0.15766851053750708",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.04821745244115927",
                    "branch:to:retriever_one": "00000000000000000000000000000004.0.5907938097782264",
                    "docs": "00000000000000000000000000000004.0.972701399851098",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.7234984738744598"
                    },
                    "rewrite_query": {
                        "branch:to:rewrite_query": "00000000000000000000000000000002.0.05597832024496252"
                    },
                    "analyzer_one": {
                        "branch:to:analyzer_one": "00000000000000000000000000000003.0.7165779439892241"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.7762711252277583"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "branch:to:retriever_one": None,
                    "docs": ["doc3", "doc4"],
                },
                "updated_channels": None,
            },
            metadata={
                "source": "loop",
                "step": 2,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2039-6354-8001-2c508c8dffd9",
                }
            },
            pending_writes=[
                ("a5602426-85f2-1fe4-c9e4-bd0127e8e53e", "docs", ["doc1", "doc2"]),
                ("a5602426-85f2-1fe4-c9e4-bd0127e8e53e", "branch:to:qa", None),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2039-6354-8001-2c508c8dffd9",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.125661+00:00",
                "id": "1f00fd5f-2039-6354-8001-2c508c8dffd9",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.6697367414225304",
                    "query": "00000000000000000000000000000003.0.04057405566428263",
                    "branch:to:rewrite_query": "00000000000000000000000000000003.0.14126716107927562",
                    "branch:to:analyzer_one": "00000000000000000000000000000003.0.7165779439892241",
                    "branch:to:retriever_two": "00000000000000000000000000000003.0.7762711252277583",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.7234984738744598"
                    },
                    "rewrite_query": {
                        "branch:to:rewrite_query": "00000000000000000000000000000002.0.05597832024496252"
                    },
                },
                "channel_values": {
                    "query": "query: what is weather in sf",
                    "branch:to:analyzer_one": None,
                    "branch:to:retriever_two": None,
                },
                "updated_channels": None,
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2038-613e-8000-ce5ebe65eb97",
                }
            },
            pending_writes=[
                (
                    "4e7cb70b-7e0f-52d0-d8aa-5439bd3f84de",
                    "query",
                    "analyzed: query: what is weather in sf",
                ),
                (
                    "4e7cb70b-7e0f-52d0-d8aa-5439bd3f84de",
                    "branch:to:retriever_one",
                    None,
                ),
                ("abcbc448-cfba-ac2b-2e39-346808f20add", "docs", ["doc3", "doc4"]),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2038-613e-8000-ce5ebe65eb97",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.125200+00:00",
                "id": "1f00fd5f-2038-613e-8000-ce5ebe65eb97",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.6697367414225304",
                    "query": "00000000000000000000000000000002.0.3399249312096154",
                    "branch:to:rewrite_query": "00000000000000000000000000000002.0.05597832024496252",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.7234984738744598"
                    },
                },
                "channel_values": {
                    "query": "what is weather in sf",
                    "branch:to:rewrite_query": None,
                },
                "updated_channels": None,
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2036-6ce4-bfff-ac42e9890362",
                }
            },
            pending_writes=[
                (
                    "d1c3a2d6-5ca2-d4c5-5217-35d86cce48a4",
                    "query",
                    "query: what is weather in sf",
                ),
                (
                    "d1c3a2d6-5ca2-d4c5-5217-35d86cce48a4",
                    "branch:to:analyzer_one",
                    None,
                ),
                (
                    "d1c3a2d6-5ca2-d4c5-5217-35d86cce48a4",
                    "branch:to:retriever_two",
                    None,
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fd5f-2036-6ce4-bfff-ac42e9890362",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T15:20:01.124678+00:00",
                "id": "1f00fd5f-2036-6ce4-bfff-ac42e9890362",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000001.0.7234984738744598"
                },
                "versions_seen": {"__input__": {}},
                "channel_values": {"__start__": {"query": "what is weather in sf"}},
                "updated_channels": None,
            },
            metadata={
                "source": "input",
                "step": -1,
                "parents": {},
            },
            parent_config=None,
            pending_writes=[
                (
                    "a9e2a749-9870-1952-0a6c-b23b6729ffda",
                    "query",
                    "what is weather in sf",
                ),
                (
                    "a9e2a749-9870-1952-0a6c-b23b6729ffda",
                    "branch:to:rewrite_query",
                    None,
                ),
            ],
        ),
    ],
    "2-start:*": [
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-515f-6b88-8004-6fffb69dd465",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.825576+00:00",
                "id": "1f00fe48-515f-6b88-8004-6fffb69dd465",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.23383372151016169",
                    "query": "00000000000000000000000000000004.0.05732679770452498",
                    "start:rewrite_query": "00000000000000000000000000000003.0.2916637829964738",
                    "rewrite_query": "00000000000000000000000000000004.0.2372002638794427",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.8860781568140047",
                    "analyzer_one": "00000000000000000000000000000005.0.648286705356163",
                    "docs": "00000000000000000000000000000005.0.19918575623485935",
                    "retriever_two": "00000000000000000000000000000005.0.46629341414062697",
                    "retriever_one": "00000000000000000000000000000006.0.9577453764095437",
                    "answer": "00000000000000000000000000000006.0.27361287406148327",
                    "qa": "00000000000000000000000000000006.0.24260043089701677",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.9575279209966122"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.3082066433110763"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.9534854313752955"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.29217346538810884"
                    },
                    "retriever_one": {
                        "analyzer_one": "00000000000000000000000000000004.0.9322215406936268"
                    },
                    "__interrupt__": {
                        "query": "00000000000000000000000000000004.0.05732679770452498",
                        "docs": "00000000000000000000000000000005.0.19918575623485935",
                        "__start__": "00000000000000000000000000000002.0.23383372151016169",
                        "rewrite_query": "00000000000000000000000000000004.0.2372002638794427",
                        "analyzer_one": "00000000000000000000000000000005.0.648286705356163",
                        "retriever_one": "00000000000000000000000000000005.0.0523757506060204",
                        "retriever_two": "00000000000000000000000000000005.0.46629341414062697",
                        "branch:to:retriever_two": "00000000000000000000000000000004.0.8860781568140047",
                        "start:rewrite_query": "00000000000000000000000000000003.0.2916637829964738",
                    },
                    "qa": {
                        "retriever_one": "00000000000000000000000000000005.0.0523757506060204"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "answer": "doc1,doc2,doc3,doc4",
                    "qa": "qa",
                },
            },
            metadata={
                "source": "loop",
                "step": 4,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-515c-679e-8003-5f85a56d5dba",
                }
            },
            pending_writes=[],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-515c-679e-8003-5f85a56d5dba",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.824251+00:00",
                "id": "1f00fe48-515c-679e-8003-5f85a56d5dba",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.23383372151016169",
                    "query": "00000000000000000000000000000004.0.05732679770452498",
                    "start:rewrite_query": "00000000000000000000000000000003.0.2916637829964738",
                    "rewrite_query": "00000000000000000000000000000004.0.2372002638794427",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.8860781568140047",
                    "analyzer_one": "00000000000000000000000000000005.0.648286705356163",
                    "docs": "00000000000000000000000000000005.0.19918575623485935",
                    "retriever_two": "00000000000000000000000000000005.0.46629341414062697",
                    "retriever_one": "00000000000000000000000000000005.0.0523757506060204",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.9575279209966122"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.3082066433110763"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.9534854313752955"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.29217346538810884"
                    },
                    "retriever_one": {
                        "analyzer_one": "00000000000000000000000000000004.0.9322215406936268"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "retriever_one": "retriever_one",
                },
            },
            metadata={
                "source": "loop",
                "step": 3,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-515b-6d12-8002-82a8f9213eae",
                }
            },
            pending_writes=[
                (
                    "4ee8637e-0a95-285e-75bc-4da721c0beab",
                    "__interrupt__",
                    [
                        Interrupt(
                            value="",
                            resumable=True,  # type: ignore[arg-type]
                            ns=["qa:4ee8637e-0a95-285e-75bc-4da721c0beab"],  # type: ignore[arg-type]
                        )
                    ],
                ),
                ("00000000-0000-0000-0000-000000000000", "__resume__", ""),
                ("4ee8637e-0a95-285e-75bc-4da721c0beab", "__resume__", [""]),
                (
                    "4ee8637e-0a95-285e-75bc-4da721c0beab",
                    "answer",
                    "doc1,doc2,doc3,doc4",
                ),
                ("4ee8637e-0a95-285e-75bc-4da721c0beab", "qa", "qa"),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-515b-6d12-8002-82a8f9213eae",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.823978+00:00",
                "id": "1f00fe48-515b-6d12-8002-82a8f9213eae",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.23383372151016169",
                    "query": "00000000000000000000000000000004.0.05732679770452498",
                    "start:rewrite_query": "00000000000000000000000000000003.0.2916637829964738",
                    "rewrite_query": "00000000000000000000000000000004.0.2372002638794427",
                    "branch:to:retriever_two": "00000000000000000000000000000004.0.8860781568140047",
                    "analyzer_one": "00000000000000000000000000000004.0.9322215406936268",
                    "docs": "00000000000000000000000000000004.0.49012772235571145",
                    "retriever_two": "00000000000000000000000000000004.0.9223450775254257",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.9575279209966122"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.3082066433110763"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.9534854313752955"
                    },
                    "retriever_two": {
                        "branch:to:retriever_two": "00000000000000000000000000000003.0.29217346538810884"
                    },
                },
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "analyzer_one": "analyzer_one",
                    "docs": ["doc3", "doc4"],
                    "retriever_two": "retriever_two",
                },
            },
            metadata={
                "source": "loop",
                "step": 2,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5059-6b30-8001-2a9ab4ca7d82",
                }
            },
            pending_writes=[
                ("16295c56-f44e-31fa-8fad-fff3f9022629", "docs", ["doc1", "doc2"]),
                (
                    "16295c56-f44e-31fa-8fad-fff3f9022629",
                    "retriever_one",
                    "retriever_one",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5059-6b30-8001-2a9ab4ca7d82",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.718258+00:00",
                "id": "1f00fe48-5059-6b30-8001-2a9ab4ca7d82",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.23383372151016169",
                    "query": "00000000000000000000000000000003.0.10748450241039154",
                    "start:rewrite_query": "00000000000000000000000000000003.0.2916637829964738",
                    "rewrite_query": "00000000000000000000000000000003.0.9534854313752955",
                    "branch:to:retriever_two": "00000000000000000000000000000003.0.29217346538810884",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.9575279209966122"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.3082066433110763"
                    },
                },
                "channel_values": {
                    "query": "query: what is weather in sf",
                    "rewrite_query": "rewrite_query",
                    "branch:to:retriever_two": "rewrite_query",
                },
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5058-6a46-8000-086bffc73797",
                }
            },
            pending_writes=[
                (
                    "baecc0e3-ea00-0e00-9436-e33cd2527faf",
                    "query",
                    "analyzed: query: what is weather in sf",
                ),
                (
                    "baecc0e3-ea00-0e00-9436-e33cd2527faf",
                    "analyzer_one",
                    "analyzer_one",
                ),
                ("96b7bfe4-269f-092c-e685-14dba6a27271", "docs", ["doc3", "doc4"]),
                (
                    "96b7bfe4-269f-092c-e685-14dba6a27271",
                    "retriever_two",
                    "retriever_two",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5058-6a46-8000-086bffc73797",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.717827+00:00",
                "id": "1f00fe48-5058-6a46-8000-086bffc73797",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.23383372151016169",
                    "query": "00000000000000000000000000000002.0.706632616485588",
                    "start:rewrite_query": "00000000000000000000000000000002.0.3082066433110763",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.9575279209966122"
                    },
                },
                "channel_values": {
                    "query": "what is weather in sf",
                    "start:rewrite_query": "__start__",
                },
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5057-62a4-bfff-1883a92a3e41",
                }
            },
            pending_writes=[
                (
                    "058cf6d6-a83c-6509-b398-5dde0b6c5773",
                    "query",
                    "query: what is weather in sf",
                ),
                (
                    "058cf6d6-a83c-6509-b398-5dde0b6c5773",
                    "rewrite_query",
                    "rewrite_query",
                ),
                (
                    "058cf6d6-a83c-6509-b398-5dde0b6c5773",
                    "branch:to:retriever_two",
                    "rewrite_query",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00fe48-5057-62a4-bfff-1883a92a3e41",
                }
            },
            checkpoint={
                "v": 2,
                "ts": "2025-04-02T17:04:20.717221+00:00",
                "id": "1f00fe48-5057-62a4-bfff-1883a92a3e41",
                "channel_versions": {
                    "__start__": "00000000000000000000000000000001.0.9575279209966122"
                },
                "versions_seen": {"__input__": {}},
                "channel_values": {"__start__": {"query": "what is weather in sf"}},
            },
            metadata={
                "source": "input",
                "step": -1,
                "parents": {},
            },
            parent_config=None,
            pending_writes=[
                (
                    "891e8564-d78f-7fb2-f15d-bce2a0ddf1c6",
                    "query",
                    "what is weather in sf",
                ),
                (
                    "891e8564-d78f-7fb2-f15d-bce2a0ddf1c6",
                    "start:rewrite_query",
                    "__start__",
                ),
            ],
        ),
    ],
    "2-quadratic": [
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-546a-64f0-8004-d2d4e06b6fb6",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.630535+00:00",
                "id": "1f00ffbe-546a-64f0-8004-d2d4e06b6fb6",
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "answer": "doc1,doc2,doc3,doc4",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "qa": "qa",
                },
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.141080617837112",
                    "query": "00000000000000000000000000000004.0.9004905790383284",
                    "start:rewrite_query": "00000000000000000000000000000003.0.013109117892399547",
                    "rewrite_query": "00000000000000000000000000000004.0.1679336326485974",
                    "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000004.0.7474512867042074",
                    "analyzer_one": "00000000000000000000000000000005.0.5817293698381076",
                    "docs": "00000000000000000000000000000005.0.9650795030435029",
                    "retriever_two": "00000000000000000000000000000005.0.77101858493518",
                    "retriever_one": "00000000000000000000000000000006.0.4984661612084784",
                    "answer": "00000000000000000000000000000006.0.6244466008661432",
                    "qa": "00000000000000000000000000000006.0.06630110662217248",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.6759219622820284"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.32002588286540445"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.32578323811902354"
                    },
                    "retriever_two": {
                        "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000003.0.8992241767805405"
                    },
                    "retriever_one": {
                        "analyzer_one": "00000000000000000000000000000004.0.2684613370070208"
                    },
                    "__interrupt__": {
                        "query": "00000000000000000000000000000004.0.9004905790383284",
                        "docs": "00000000000000000000000000000005.0.9650795030435029",
                        "__start__": "00000000000000000000000000000002.0.141080617837112",
                        "rewrite_query": "00000000000000000000000000000004.0.1679336326485974",
                        "analyzer_one": "00000000000000000000000000000005.0.5817293698381076",
                        "retriever_one": "00000000000000000000000000000005.0.222301724202566",
                        "retriever_two": "00000000000000000000000000000005.0.77101858493518",
                        "start:rewrite_query": "00000000000000000000000000000003.0.013109117892399547",
                        "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000004.0.7474512867042074",
                    },
                    "qa": {
                        "retriever_one": "00000000000000000000000000000005.0.222301724202566"
                    },
                },
            },
            metadata={
                "source": "loop",
                "step": 4,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5466-61ac-8003-7ec684cd12cc",
                }
            },
            pending_writes=[],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5466-61ac-8003-7ec684cd12cc",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.628817+00:00",
                "id": "1f00ffbe-5466-61ac-8003-7ec684cd12cc",
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                    "retriever_one": "retriever_one",
                },
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.141080617837112",
                    "query": "00000000000000000000000000000004.0.9004905790383284",
                    "start:rewrite_query": "00000000000000000000000000000003.0.013109117892399547",
                    "rewrite_query": "00000000000000000000000000000004.0.1679336326485974",
                    "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000004.0.7474512867042074",
                    "analyzer_one": "00000000000000000000000000000005.0.5817293698381076",
                    "docs": "00000000000000000000000000000005.0.9650795030435029",
                    "retriever_two": "00000000000000000000000000000005.0.77101858493518",
                    "retriever_one": "00000000000000000000000000000005.0.222301724202566",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.6759219622820284"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.32002588286540445"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.32578323811902354"
                    },
                    "retriever_two": {
                        "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000003.0.8992241767805405"
                    },
                    "retriever_one": {
                        "analyzer_one": "00000000000000000000000000000004.0.2684613370070208"
                    },
                },
            },
            metadata={
                "source": "loop",
                "step": 3,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5465-6248-8002-ee1d8bdbbee5",
                }
            },
            pending_writes=[
                (
                    "369e94b1-77d1-d67a-ab59-23d1ba20ee73",
                    "__interrupt__",
                    [
                        Interrupt(
                            value="",
                            resumable=True,
                            ns=["qa:369e94b1-77d1-d67a-ab59-23d1ba20ee73"],  # type: ignore[arg-type]
                        )
                    ],
                ),
                ("00000000-0000-0000-0000-000000000000", "__resume__", ""),
                ("369e94b1-77d1-d67a-ab59-23d1ba20ee73", "__resume__", [""]),
                (
                    "369e94b1-77d1-d67a-ab59-23d1ba20ee73",
                    "answer",
                    "doc1,doc2,doc3,doc4",
                ),
                ("369e94b1-77d1-d67a-ab59-23d1ba20ee73", "qa", "qa"),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5465-6248-8002-ee1d8bdbbee5",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.628408+00:00",
                "id": "1f00ffbe-5465-6248-8002-ee1d8bdbbee5",
                "channel_values": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc3", "doc4"],
                    "analyzer_one": "analyzer_one",
                    "retriever_two": "retriever_two",
                },
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.141080617837112",
                    "query": "00000000000000000000000000000004.0.9004905790383284",
                    "start:rewrite_query": "00000000000000000000000000000003.0.013109117892399547",
                    "rewrite_query": "00000000000000000000000000000004.0.1679336326485974",
                    "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000004.0.7474512867042074",
                    "analyzer_one": "00000000000000000000000000000004.0.2684613370070208",
                    "docs": "00000000000000000000000000000004.0.37458911821520957",
                    "retriever_two": "00000000000000000000000000000004.0.7340649978617967",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.6759219622820284"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.32002588286540445"
                    },
                    "analyzer_one": {
                        "rewrite_query": "00000000000000000000000000000003.0.32578323811902354"
                    },
                    "retriever_two": {
                        "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000003.0.8992241767805405"
                    },
                },
            },
            metadata={
                "source": "loop",
                "step": 2,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-536a-6dca-8001-c7e021a73244",
                }
            },
            pending_writes=[
                ("601f2099-cb23-41f9-ae64-9b4e4a6b675e", "docs", ["doc1", "doc2"]),
                (
                    "601f2099-cb23-41f9-ae64-9b4e4a6b675e",
                    "retriever_one",
                    "retriever_one",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-536a-6dca-8001-c7e021a73244",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.525915+00:00",
                "id": "1f00ffbe-536a-6dca-8001-c7e021a73244",
                "channel_values": {
                    "query": "query: what is weather in sf",
                    "rewrite_query": "rewrite_query",
                    "branch:rewrite_query:rewrite_query_then:retriever_two": "rewrite_query",
                },
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.141080617837112",
                    "query": "00000000000000000000000000000003.0.8982471206042032",
                    "start:rewrite_query": "00000000000000000000000000000003.0.013109117892399547",
                    "rewrite_query": "00000000000000000000000000000003.0.32578323811902354",
                    "branch:rewrite_query:rewrite_query_then:retriever_two": "00000000000000000000000000000003.0.8992241767805405",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.6759219622820284"
                    },
                    "rewrite_query": {
                        "start:rewrite_query": "00000000000000000000000000000002.0.32002588286540445"
                    },
                },
            },
            metadata={
                "source": "loop",
                "step": 1,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5369-6de4-8000-70bd3810f0ca",
                }
            },
            pending_writes=[
                (
                    "88757475-bc8c-934c-7d18-921cb2d94864",
                    "query",
                    "analyzed: query: what is weather in sf",
                ),
                (
                    "88757475-bc8c-934c-7d18-921cb2d94864",
                    "analyzer_one",
                    "analyzer_one",
                ),
                ("53c60600-588b-e49e-a9d2-bbcbf30a7497", "docs", ["doc3", "doc4"]),
                (
                    "53c60600-588b-e49e-a9d2-bbcbf30a7497",
                    "retriever_two",
                    "retriever_two",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5369-6de4-8000-70bd3810f0ca",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.525508+00:00",
                "id": "1f00ffbe-5369-6de4-8000-70bd3810f0ca",
                "channel_values": {
                    "query": "what is weather in sf",
                    "start:rewrite_query": "__start__",
                },
                "channel_versions": {
                    "__start__": "00000000000000000000000000000002.0.141080617837112",
                    "query": "00000000000000000000000000000002.0.06948551802156189",
                    "start:rewrite_query": "00000000000000000000000000000002.0.32002588286540445",
                },
                "versions_seen": {
                    "__input__": {},
                    "__start__": {
                        "__start__": "00000000000000000000000000000001.0.6759219622820284"
                    },
                },
            },
            metadata={
                "source": "loop",
                "step": 0,
                "parents": {},
            },
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5368-6c0a-bfff-ac22bea4c512",
                }
            },
            pending_writes=[
                (
                    "3bb470dd-9bf8-6216-b5fb-e50162991da1",
                    "query",
                    "query: what is weather in sf",
                ),
                (
                    "3bb470dd-9bf8-6216-b5fb-e50162991da1",
                    "rewrite_query",
                    "rewrite_query",
                ),
                (
                    "3bb470dd-9bf8-6216-b5fb-e50162991da1",
                    "branch:rewrite_query:rewrite_query_then:retriever_two",
                    "rewrite_query",
                ),
            ],
        ),
        CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "1f00ffbe-5368-6c0a-bfff-ac22bea4c512",
                }
            },
            checkpoint={
                "v": 1,
                "ts": "2025-04-02T19:51:40.525051+00:00",
                "id": "1f00ffbe-5368-6c0a-bfff-ac22bea4c512",
                "channel_values": {"__start__": {"query": "what is weather in sf"}},
                "channel_versions": {
                    "__start__": "00000000000000000000000000000001.0.6759219622820284"
                },
                "versions_seen": {"__input__": {}},
            },
            metadata={
                "source": "input",
                "step": -1,
                "parents": {},
            },
            parent_config=None,
            pending_writes=[
                (
                    "aa8c5e8a-da6f-ccb1-f8a9-3b145cdfe7a4",
                    "query",
                    "what is weather in sf",
                ),
                (
                    "aa8c5e8a-da6f-ccb1-f8a9-3b145cdfe7a4",
                    "start:rewrite_query",
                    "__start__",
                ),
            ],
        ),
    ],
}


def make_state_graph() -> StateGraph:
    def sorted_add(x: list[str], y: list[str] | list[tuple[str, str]]) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        interrupt("")
        return {"answer": ",".join(data["docs"])}

    def rewrite_query_then(data: State) -> Literal["retriever_two"]:
        return "retriever_two"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    workflow.add_edge("retriever_one", "qa")
    workflow.set_finish_point("qa")
    return workflow


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("source,target", [("2-start:*", "3"), ("2-quadratic", "3")])
def test_migrate_checkpoints(source: str, target: str) -> None:
    # Check that the migration function works as expected
    builder = make_state_graph()
    graph = builder.compile()

    source_checkpoints = list(reversed(SAVED_CHECKPOINTS[source]))
    target_checkpoints = list(reversed(SAVED_CHECKPOINTS[target]))
    assert len(source_checkpoints) == len(target_checkpoints)
    for idx, (source_checkpoint, target_checkpoint) in enumerate(
        zip(source_checkpoints, target_checkpoints)
    ):
        # copy the checkpoint to avoid modifying the original
        migrated = copy_checkpoint(source_checkpoint.checkpoint)
        # migrate the checkpoint
        graph._migrate_checkpoint(migrated)
        # replace values that don't need to match exactly
        migrated["id"] = AnyStr()
        migrated["ts"] = AnyStr()
        migrated["v"] = AnyInt()
        for k in migrated["channel_values"]:
            migrated["channel_values"][k] = AnyObject()
        for v in migrated["channel_versions"]:
            migrated["channel_versions"][v] = AnyStr(
                migrated["channel_versions"][v].split(".")[0]
            )
        for c in migrated["versions_seen"]:
            for v in migrated["versions_seen"][c]:
                migrated["versions_seen"][c][v] = AnyStr(
                    migrated["versions_seen"][c][v].split(".")[0]
                )
        # check that the migrated checkpoint matches the target checkpoint
        assert migrated == target_checkpoint.checkpoint, (
            f"Checkpoint mismatch at index {idx}"
        )


@NEEDS_CONTEXTVARS
def test_latest_checkpoint_state_graph(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    builder = make_state_graph()
    app = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert [
        *app.stream({"query": "what is weather in sf"}, config, durability="async")
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {
            "__interrupt__": (
                Interrupt(
                    value="",
                    id=AnyStr(),
                ),
            )
        },
    ]

    assert [*app.stream(Command(resume=""), config, durability="async")] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    # check history with current checkpoints matches expected history
    history = [*app.get_state_history(config)]
    expected_history = get_expected_history()
    assert len(history) == len(expected_history)
    assert history[0] == expected_history[0]
    assert history[1] == expected_history[1]
    assert history[2] == expected_history[2]
    assert history[3] == expected_history[3]
    assert history[4] == expected_history[4]
    assert history[5] == expected_history[5]


@NEEDS_CONTEXTVARS
async def test_latest_checkpoint_state_graph_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    builder = make_state_graph()
    app = builder.compile(checkpointer=async_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        async for c in app.astream(
            {"query": "what is weather in sf"}, config, durability="async"
        )
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {
            "__interrupt__": (
                Interrupt(
                    value="",
                    id=AnyStr(),
                ),
            )
        },
    ]

    assert [
        c async for c in app.astream(Command(resume=""), config, durability="async")
    ] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    # check history with current checkpoints matches expected history
    history = [c async for c in app.aget_state_history(config)]
    expected_history = get_expected_history()
    assert len(history) == len(expected_history)
    assert history[0] == expected_history[0]
    assert history[1] == expected_history[1]
    assert history[2] == expected_history[2]
    assert history[3] == expected_history[3]
    assert history[4] == expected_history[4]
    assert history[5] == expected_history[5]


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("checkpoint_version", ["3", "2-start:*", "2-quadratic"])
def test_saved_checkpoint_state_graph(
    sync_checkpointer: BaseCheckpointSaver,
    checkpoint_version: str,
) -> None:
    builder = make_state_graph()
    app = builder.compile(checkpointer=sync_checkpointer)

    thread1 = "1"
    config = {"configurable": {"thread_id": thread1, "checkpoint_ns": ""}}

    # save checkpoints
    parent_id: str | None = None
    for checkpoint in reversed(SAVED_CHECKPOINTS[checkpoint_version]):
        grouped_writes = defaultdict(list)
        for write in checkpoint.pending_writes:
            grouped_writes[write[0]].append(write[1:])
        for tid, group in grouped_writes.items():
            sync_checkpointer.put_writes(checkpoint.config, group, tid)
        sync_checkpointer.put(
            patch_configurable(config, {"checkpoint_id": parent_id}),
            checkpoint.checkpoint,
            checkpoint.metadata,
            checkpoint.checkpoint["channel_versions"],
        )
        parent_id = checkpoint.checkpoint["id"]

    # load history
    history = [*app.get_state_history(config)]
    # check history with saved checkpoints matches expected history
    exc_task_results: int = 0
    if checkpoint_version == "2-start:*":
        exc_task_results = 1
    elif checkpoint_version == "2-quadratic":
        exc_task_results = 2
    expected_history = get_expected_history(exc_task_results=exc_task_results)
    assert len(history) == len(expected_history)
    assert history[0] == expected_history[0]
    assert history[1] == expected_history[1]
    assert history[2] == expected_history[2]
    assert history[3] == expected_history[3]
    assert history[4] == expected_history[4]
    assert history[5] == expected_history[5]

    # resume from 2nd to latest checkpoint
    assert [*app.stream(Command(resume=""), history[1].config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]
    # new checkpoint should match the latest checkpoint in history
    latest_state = app.get_state(config)
    assert (
        StateSnapshot(
            values=latest_state.values,
            next=latest_state.next,
            config=patch_configurable(latest_state.config, {"checkpoint_id": AnyStr()}),
            metadata=AnyDict(latest_state.metadata),
            created_at=AnyStr(),
            parent_config=latest_state.parent_config,
            tasks=latest_state.tasks,
            interrupts=latest_state.interrupts,
        )
        == history[0]
    )


@NEEDS_CONTEXTVARS
@pytest.mark.parametrize("checkpoint_version", ["3", "2-start:*", "2-quadratic"])
async def test_saved_checkpoint_state_graph_async(
    async_checkpointer: BaseCheckpointSaver,
    checkpoint_version: str,
) -> None:
    builder = make_state_graph()
    app = builder.compile(checkpointer=async_checkpointer)

    thread1 = "1"
    config = {"configurable": {"thread_id": thread1, "checkpoint_ns": ""}}

    # save checkpoints
    parent_id: str | None = None
    for checkpoint in reversed(SAVED_CHECKPOINTS[checkpoint_version]):
        grouped_writes = defaultdict(list)
        for write in checkpoint.pending_writes:
            grouped_writes[write[0]].append(write[1:])
        for tid, group in grouped_writes.items():
            await async_checkpointer.aput_writes(checkpoint.config, group, tid)
        await async_checkpointer.aput(
            patch_configurable(config, {"checkpoint_id": parent_id}),
            checkpoint.checkpoint,
            checkpoint.metadata,
            checkpoint.checkpoint["channel_versions"],
        )
        parent_id = checkpoint.checkpoint["id"]

    # load history
    history = [c async for c in app.aget_state_history(config)]
    # check history with saved checkpoints matches expected history
    exc_task_results: int = 0
    if checkpoint_version == "2-start:*":
        exc_task_results = 1
    elif checkpoint_version == "2-quadratic":
        exc_task_results = 2
    expected_history = get_expected_history(exc_task_results=exc_task_results)
    assert len(history) == len(expected_history)
    assert history[0] == expected_history[0]
    assert history[1] == expected_history[1]
    assert history[2] == expected_history[2]
    assert history[3] == expected_history[3]
    assert history[4] == expected_history[4]
    assert history[5] == expected_history[5]

    # resume from 2nd to latest checkpoint
    assert [c async for c in app.astream(Command(resume=""), history[1].config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]
    # new checkpoint should match the latest checkpoint in history
    latest_state = await app.aget_state(config)
    assert (
        StateSnapshot(
            values=latest_state.values,
            next=latest_state.next,
            config=patch_configurable(latest_state.config, {"checkpoint_id": AnyStr()}),
            metadata=AnyDict(latest_state.metadata),
            created_at=AnyStr(),
            parent_config=latest_state.parent_config,
            tasks=latest_state.tasks,
            interrupts=latest_state.interrupts,
        )
        == history[0]
    )
