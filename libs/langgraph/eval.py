from collections import defaultdict
from typing import Any, Dict, Iterable, List, TypedDict

from langchain_core.runnables import RunnableConfig

from langgraph.pregel import Pregel
from langgraph.pregel.types import StateSnapshot


class TestCase(TypedDict):
    id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]


def _node_test_cases(snapshots: Iterable[StateSnapshot]) -> Dict[str, List[TestCase]]:
    test_cases = defaultdict(list)
    partials: Dict[str, Dict[str, TestCase]] = defaultdict(dict)
    for snapshot in snapshots:
        thread_ts = snapshot.config["configurable"]["thread_ts"]
        if partials[thread_ts]:
            for node, partial in partials[thread_ts].items():
                test_cases[node].append(
                    {
                        "id": partial["id"],
                        "inputs": snapshot.values,
                        "outputs": partial["outputs"],
                        "metadata": partial["metadata"],
                    }
                )
            partials[thread_ts].clear()
        if (
            (writes := snapshot.metadata["writes"])
            and snapshot.parent_config
            and isinstance(writes, dict)
            and snapshot.metadata["source"] == "loop"
        ):
            parent_thread_ts = snapshot.parent_config["configurable"]["thread_ts"]
            for node, outputs in writes.items():
                partials[parent_thread_ts][node] = {
                    "id": snapshot.config["configurable"]["thread_ts"],
                    "inputs": None,
                    "outputs": outputs,
                    "metadata": {
                        "source": snapshot.metadata["source"],
                        "step": snapshot.metadata["step"],
                        **snapshot.config["configurable"],
                    },
                }
    return dict(test_cases)


def extract_node_test_cases_from_thread(
    graph: Pregel, config: RunnableConfig
) -> Dict[str, List[TestCase]]:
    return _node_test_cases(graph.get_state_history(config))


async def aextract_node_test_cases_from_thread(
    graph: Pregel, config: RunnableConfig
) -> Dict[str, List[TestCase]]:
    return _node_test_cases([s async for s in graph.get_state_history(config)])


def _graph_test_case(snapshots: Iterable[StateSnapshot]) -> TestCase:
    test_case = TestCase(
        id=None,
        inputs={
            "input": [],
        },
        outputs={
            "output": [],
            "steps": [],
        },
    )
    is_acc_steps = False
    for snapshot in snapshots:
        if not test_case["id"]:
            test_case["id"] = snapshot.config["configurable"]["thread_id"]
        if not snapshot.next:
            is_acc_steps = True
            test_case["outputs"]["output"].append(snapshot.values)
            test_case["outputs"]["steps"].append([])
            if not test_case.get("metadata"):
                test_case["metadata"] = snapshot.config["configurable"]
        if (
            is_acc_steps
            and snapshot.metadata["source"] == "loop"
            and snapshot.metadata["writes"]
        ):
            for node in snapshot.metadata["writes"]:
                test_case["outputs"]["steps"][-1].append(node)
        if is_acc_steps and snapshot.metadata["source"] == "input":
            test_case["inputs"]["input"].append(snapshot.metadata["writes"])
    test_case["inputs"]["input"].reverse()
    test_case["outputs"]["output"].reverse()
    test_case["outputs"]["steps"].reverse()
    for ss in test_case["outputs"]["steps"]:
        ss.reverse()
    return test_case


def extract_graph_test_case_from_thread(
    graph: Pregel, config: RunnableConfig
) -> TestCase:
    return _graph_test_case(graph.get_state_history(config))


async def aextract_graph_test_case_from_thread(
    graph: Pregel, config: RunnableConfig
) -> TestCase:
    return _graph_test_case([s async for s in graph.get_state_history(config)])
