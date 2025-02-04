# How to use Pydantic model as graph state

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/#state">
                    State
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes">
                    Nodes
                </a>
            </li>
            <li>
                <a href="https://github.com/pydantic/pydantic">
                    Pydantic
                </a>: this is a popular Python library for run time validation.
            </li>
        </ul>
    </p>
</div>

A [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph) accepts a `state_schema` argument on initialization that specifies the "shape" of the state that the nodes in the graph can access and update.

In our examples, we typically use a python-native `TypedDict` for `state_schema` (or in the case of [MessageGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#messagegraph), a [list](https://docs.python.org/3/library/stdtypes.html#list)), but `state_schema` can be any [type](https://docs.python.org/3/library/stdtypes.html#type-objects).

In this how-to guide, we'll see how a [Pydantic BaseModel](https://docs.pydantic.dev/latest/api/base_model/). can be used for `state_schema` to add run time validation on **inputs**.


<div class="admonition note">
    <p class="admonition-title">Known Limitations</p>
    <p>
        <ul>
            <li>
              This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.      
            </li>        
            <li>
                Currently, the `output` of the graph will **NOT** be an instance of a pydantic model.
            </li>
            <li>
                Run-time validation only occurs on **inputs** into nodes, not on the outputs.
            </li>
            <li>
                The validation error trace from pydantic does not show which node the error arises in.
            </li>
        </ul>
    </p>
</div>

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Input Validation


```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel


# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str


def node(state: OverallState):
    return {"a": "goodbye"}


# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(node)  # node_1 is the first node
builder.add_edge(START, "node")  # Start the graph with node_1
builder.add_edge("node", END)  # End the graph after node_1
graph = builder.compile()

# Test the graph with a valid input
graph.invoke({"a": "hello"})
```






Invoke the graph with an **invalid** input


```python
try:
    graph.invoke({"a": 123})  # Should be a string
except Exception as e:
    print("An exception was raised because `a` is an integer rather than a string.")
    print(e)
```

## Multiple Nodes

Run-time validation will also work in a multi-node graph. In the example below `bad_node` updates `a` to an integer. 

Because run-time validation occurs on **inputs**, the validation error will occur when `ok_node` is called (not when `bad_node` returns an update to the state which is inconsistent with the schema).


```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel


# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str


def bad_node(state: OverallState):
    return {
        "a": 123  # Invalid
    }


def ok_node(state: OverallState):
    return {"a": "goodbye"}


# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(bad_node)
builder.add_node(ok_node)
builder.add_edge(START, "bad_node")
builder.add_edge("bad_node", "ok_node")
builder.add_edge("ok_node", END)
graph = builder.compile()

# Test the graph with a valid input
try:
    graph.invoke({"a": "hello"})
except Exception as e:
    print("An exception was raised because bad_node sets `a` to an integer.")
    print(e)
```
