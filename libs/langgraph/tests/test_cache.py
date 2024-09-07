import hashlib
import json
import operator
import time

from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from langgraph.checkpoint.base import BaseCheckpointSaver
import pytest
from langchain_core.runnables import (
    RunnableConfig,
)

from syrupy import SnapshotAssertion
from langgraph.graph.state import StateGraph
from langgraph.pregel.types import CachePolicy


def custom_cache_key(input: Any, config: Optional[RunnableConfig] = None) -> str:
    """
    Generate a cache key based on the input and config.

    Args:
        input (Any): The input to the node.
        config (Optional[RunnableConfig]): The configuration for the node.

    Returns:
        str: A string key under which the output should be cached.
    """
    # Convert input to a JSON-serializable format
    if isinstance(input, dict):
        input_str = json.dumps(input, sort_keys=True)
    elif isinstance(input, (str, int, float, bool)):
        input_str = str(input)
    else:
        input_str = str(hash(input))

    # Extract relevant parts from the config
    config_str = ""
    if config:
        relevant_config = {
            "tags": config.get("tags", []),
            "metadata": config.get("metadata", {}),
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
    # Combine input and config strings
    combined_str = f"{input_str}|{config_str}"
    # Generate a hash of the combined string
    return hashlib.md5(combined_str.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("checkpointer_name", ["postgres"])
def test_in_one_fan_out_state_graph_waiting_edge_via_branch_with_cache(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    workflow = StateGraph(State)
    call_count = 0

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    def retriever_one(data: State) -> State:
        nonlocal call_count
        call_count += 1
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    def rewrite_query_then(data: State) -> Literal["retriever_two"]:
        return "retriever_two"

    config = RunnableConfig(configurable={"thread_id": "1"})

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node(
        "retriever_one",
        retriever_one,
        cache=CachePolicy(custom_cache_key({"user_id": "a user"}, config)),
    )
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app_with_checkpointer = workflow.compile(
        checkpointer=checkpointer,
    )

    interrupt_results = [
        c
        for c in app_with_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]
    assert interrupt_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},  # This item is extra
    ]
    assert call_count == 1
    config = RunnableConfig(configurable={"thread_id": "2"})

    stream_results = [
        c
        for c in app_with_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]

    assert stream_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}, "__metadata__": {"cached": True}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    # Should not increase count because of cache
    assert call_count == 1

    config = RunnableConfig(configurable={"thread_id": "3"})

    stream_results = [
        c
        for c in app_with_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]

    assert stream_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}, "__metadata__": {"cached": True}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    # Should not increase count because of cache
    assert call_count == 1

    # Cache is not used when checkpointer is not provided
    app_without_checkpointer = workflow.compile()
    interrupt_results = [
        c
        for c in app_without_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]
    assert interrupt_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},  # This item is extra
    ]
    assert call_count == 2

    # Test a new workflow with the same cache key
    new_workflow = StateGraph(State)
    config = RunnableConfig(configurable={"thread_id": "4"})
    new_workflow.add_node("rewrite_query", rewrite_query)
    new_workflow.add_node("analyzer_one", analyzer_one)
    new_workflow.add_node(
        "retriever_one",
        retriever_one,
        cache=CachePolicy(custom_cache_key({"user_id": "a user"}, config)),
    )
    new_workflow.add_node("retriever_two", retriever_two)
    new_workflow.add_node("qa", qa)

    new_workflow.set_entry_point("rewrite_query")
    new_workflow.add_edge("rewrite_query", "analyzer_one")
    new_workflow.add_edge("analyzer_one", "retriever_one")
    new_workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    new_workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    new_workflow.set_finish_point("qa")

    app_with_checkpointer = new_workflow.compile(
        checkpointer=checkpointer,
    )

    interrupt_results = [
        c
        for c in app_with_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]
    assert interrupt_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}, "__metadata__": {"cached": True}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},  # This item is extra
    ]
    assert call_count == 2

    # Test a new workflow with a different cache key
    another_new_workflow = StateGraph(State)
    config = RunnableConfig(configurable={"thread_id": "5"})
    another_new_workflow.add_node("rewrite_query", rewrite_query)
    another_new_workflow.add_node("analyzer_one", analyzer_one)
    another_new_workflow.add_node(
        "retriever_one",
        retriever_one,
        cache=CachePolicy(custom_cache_key({"user_id": "a different user"}, config)),
    )
    another_new_workflow.add_node("retriever_two", retriever_two)
    another_new_workflow.add_node("qa", qa)

    another_new_workflow.set_entry_point("rewrite_query")
    another_new_workflow.add_edge("rewrite_query", "analyzer_one")
    another_new_workflow.add_edge("analyzer_one", "retriever_one")
    another_new_workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    another_new_workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    another_new_workflow.set_finish_point("qa")

    app_with_checkpointer = another_new_workflow.compile(
        checkpointer=checkpointer,
    )

    interrupt_results = [
        c
        for c in app_with_checkpointer.stream(
            {"query": "what is weather in sf"}, config
        )
    ]
    assert interrupt_results == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},  # This item is extra
    ]
    assert call_count == 3
