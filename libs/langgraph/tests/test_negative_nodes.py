"""Streamlined edge case tests - no compilation"""
import sys
sys.path.insert(0, '.')

from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def dummy_node(state: State, config: RunnableConfig | None = None) -> State:
    return {"value": state["value"] + 1}

tests_passed = 0
tests_failed = 0

def test(name, fn):
    global tests_passed, tests_failed
    print(f"\n[{name}]", end=" ")
    try:
        fn()
        print("✓ PASS")
        tests_passed += 1
    except AssertionError as e:
        print(f"✗ FAIL: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"✗ FAIL (Unexpected): {e}")
        tests_failed += 1

print("=" * 70)
print("EDGE CASE TESTS FOR NEGATIVE NODE")
print("=" * 70)

# Test 1
def t1():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_edge("a", "b")
test("Normal node edge", t1)

# Test 2
def t2():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    builder.add_edge(START, "a")
test("START to node", t2)

# Test 3
def t3():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    builder.add_edge("a", END)
test("Node to END", t3)

# Test 4
def t4():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    try:
        builder.add_edge("a", "b")
        raise AssertionError("Should reject negative node without prob distribution")
    except ValueError as e:
        if "probabilistic distribution" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Negative without prob (reject)", t4)

# Test 5
def t5():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.5, 0.5])
test("Negative with prob (multiple)", t5)

# Test 6
def t6():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    try:
        builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.3, 0.5])
        raise AssertionError("Should reject invalid prob sum")
    except ValueError as e:
        if "sum to 1.0" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Negative invalid prob sum (reject)", t6)

# Test 7
def t7():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    try:
        builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[-0.2, 1.2])
        raise AssertionError("Should reject negative probabilities")
    except ValueError as e:
        if "non-negative" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Negative with negative probs (reject)", t7)

# Test 8
def t8():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    try:
        builder.add_edge("a", ["b", "c"], nodes_prob_distribution=[0.5])
        raise AssertionError("Should reject mismatched length")
    except ValueError as e:
        if "Length" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Negative mismatched length (reject)", t8)

# Test 9
def t9():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    try:
        builder.add_edge("a", ["b", "c"])
        raise AssertionError("Should reject multiple edges from regular node")
    except ValueError as e:
        if "Only negative nodes" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Regular with multiple edges (reject)", t9)

# Test 10
def t10():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_edge("a", ["b", END], nodes_prob_distribution=[0.6, 0.4])
test("Negative to END with prob", t10)

# Test 11
def t11():
    builder = StateGraph(State)
    builder.add_negative_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    try:
        builder.add_edge(["a", "b"], "c")
        raise AssertionError("Should reject multiple starts with negative node")
    except ValueError as e:
        if "Cannot have multiple start nodes" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Multiple starts with negative (reject)", t11)

# Test 12
def t12():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    builder.add_node("b", dummy_node)
    builder.add_node("c", dummy_node)
    builder.add_edge(["a", "b"], "c")
test("Multiple regular starts", t12)

# Test 13
def t13():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    try:
        builder.add_edge("nonexistent", "a")
        raise AssertionError("Should reject nonexistent start node")
    except ValueError as e:
        if "Need to add_node" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Nonexistent start (reject)", t13)

# Test 14
def t14():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    try:
        builder.add_edge("a", "nonexistent")
        raise AssertionError("Should reject nonexistent end node")
    except ValueError as e:
        if "Need to add_node" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("Nonexistent end (reject)", t14)

# Test 15
def t15():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    try:
        builder.add_edge(END, "a")
        raise AssertionError("Should reject END as start")
    except ValueError as e:
        if "END cannot be a start" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("END as start (reject)", t15)

# Test 16
def t16():
    builder = StateGraph(State)
    builder.add_node("a", dummy_node)
    try:
        builder.add_edge("a", START)
        raise AssertionError("Should reject START as end")
    except ValueError as e:
        if "START cannot be an end" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
test("START as end (reject)", t16)

print("\n" + "=" * 70)
print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
print("=" * 70)
