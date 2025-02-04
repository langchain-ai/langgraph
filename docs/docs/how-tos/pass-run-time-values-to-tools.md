# How to pass runtime values to tools

Sometimes, you want to let a tool-calling LLM populate a *subset* of the tool functions' arguments and provide the other values for the other arguments at runtime. If you're using LangChain-style [tools](https://python.langchain.com/docs/concepts/#tools), an easy way to handle this is by annotating function parameters with [InjectedArg](https://python.langchain.com/docs/how_to/tool_runtime/). This annotation excludes that parameter from being shown to the LLM.

In LangGraph applications you might want to pass the graph state or [shared memory](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/) (store) to the tools at runtime. This type of stateful tools is useful when a tool's output is affected by past agent steps (e.g. if you're using a sub-agent as a tool, and want to pass the message history in to the sub-agent), or when a tool's input needs to be validated given context from past agent steps.

In this guide we'll demonstrate how to do so using LangGraph's prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/).

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide targets **LangChain tool calling** assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#tools">
                    Tools
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/#state">
                    State
                </a>
            </li>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent">
                    Tool-calling
                </a>
            </li>
        </ul>
        You can still use tool calling in LangGraph using your provider SDK without losing any of LangGraph's core features.
    </p>
</div> 

The core technique the examples below is to **annotate** a parameter as "injected", meaning it will be injected by your program and should not be seen or populated by the LLM. Let the following codesnippet serve as a tl;dr:

```python
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore

from langgraph.prebuilt import InjectedState, InjectedStore


# Can be sync or async; @tool decorator not required
async def my_tool(
    # These arguments are populated by the LLM
    some_arg: str,
    another_arg: float,
    # The config: RunnableConfig is always available in LangChain calls
    # This is not exposed to the LLM
    config: RunnableConfig,
    # The following three are specific to the prebuilt ToolNode
    # (and `create_react_agent` by extension). If you are invoking the
    # tool on its own (in your own node), then you would need to provide these yourself.
    store: Annotated[BaseStore, InjectedStore],
    # This passes in the full state.
    state: Annotated[State, InjectedState],
    # You can also inject single fields from your state if you
    messages: Annotated[list, InjectedState("messages")]
    # The following is not compatible with create_react_agent or ToolNode
    # You can also exclude other arguments from being shown to the model.
    # These must be provided manually and are useful if you call the tools/functions in your own node
    # some_other_arg=Annotated["MyPrivateClass", InjectedToolArg],
):
    """Call my_tool to have an impact on the real world.

    Args:
        some_arg: a very important argument
        another_arg: another argument the LLM will provide
    """ # The docstring becomes the description for your tool and is passed to the model
    print(some_arg, another_arg, config, store, state, messages)
    # Config, some_other_rag, store, and state  are all "hidden" from
    # LangChain models when passed to bind_tools or with_structured_output
    return "... some response"
```


```python

```

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain-openai
```

Next, we need to set API keys for OpenAI (the chat model we will use).


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

## Pass graph state to tools

Let's first take a look at how to give our tools access to the graph state. We'll need to define our graph state:


```python
from typing import List

# this is the state schema used by the prebuilt create_react_agent we'll be using below
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.documents import Document


class State(AgentState):
    docs: List[str]
```

### Define the tools

We'll want our tool to take graph state as an input, but we don't want the model to try to generate this input when calling the tool. We can use the `InjectedState` annotation to mark arguments as required graph state (or some field of graph state. These arguments will not be generated by the model. When using `ToolNode`, graph state will automatically be passed in to the relevant tools and arguments.

In this example we'll create a tool that returns Documents and then another tool that actually cites the Documents that justify a claim.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>  


```python
from typing import List, Tuple
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState


@tool
def get_context(question: str, state: Annotated[dict, InjectedState]):
    """Get relevant context for answering the question."""
    return "\n\n".join(doc for doc in state["docs"])
```

If we look at the input schemas for these tools, we'll see that `state` is still listed:


```python
get_context.get_input_schema().schema()
```






But if we look at the tool call schema, which is what is passed to the model for tool-calling, `state` has been removed:


```python
get_context.tool_call_schema.schema()
```






### Define the graph

In this example we will be using a [prebuilt ReAct agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/). We'll first need to define our model and a tool-calling node ([ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)):


```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_context]

# ToolNode will automatically take care of injecting state into tools
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
graph = create_react_agent(model, tools, state_schema=State, checkpointer=checkpointer)
```

### Use it!


```python
docs = [
    "FooBar company just raised 1 Billion dollars!",
    "FooBar company was founded in 2019",
]

inputs = {
    "messages": [{"type": "user", "content": "what's the latest news about FooBar"}],
    "docs": docs,
}
config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream(inputs, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

## Pass shared memory (store) to the graph

You might also want to give tools access to memory that is shared across multiple conversations or users. We can do it by passing LangGraph [Store](https://langchain-ai.github.io/langgraph/how-tos/cross-thread-persistence/) to the tools using a different annotation -- `InjectedStore`.

Let's modify our example to save the documents in an in-memory store and retrieve them using `get_context` tool. We'll also make the documents accessible based on a user ID, so that some documents are only visible to certain users. The tool will then use the `user_id` provided in the [config](https://langchain-ai.github.io/langgraph/how-tos/pass-config-to-tools/) to retrieve a correct set of documents.

<div class="admonition note">
    <p class="admonition-title">Note</p>
    <list>
        <li>
        Support for <code>Store</code> API and <code>InjectedStore</code> used in this notebook was added in LangGraph <code>v0.2.34</code>.
        </li>
        <li>
        <code>InjectedStore</code> annotation requires <code>langchain-core >= 0.3.8</code>
        </li>
    <list>
</div>


```python
from langgraph.store.memory import InMemoryStore

doc_store = InMemoryStore()

namespace = ("documents", "1")  # user ID
doc_store.put(
    namespace, "doc_0", {"doc": "FooBar company just raised 1 Billion dollars!"}
)
namespace = ("documents", "2")  # user ID
doc_store.put(namespace, "doc_1", {"doc": "FooBar company was founded in 2019"})
```

### Define the tools


```python
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore


@tool
def get_context(
    question: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> Tuple[str, List[Document]]:
    """Get relevant context for answering the question."""
    user_id = config.get("configurable", {}).get("user_id")
    docs = [item.value["doc"] for item in store.search(("documents", user_id))]
    return "\n\n".join(doc for doc in docs)
```

We can also verify that the tool-calling model will ignore `store` arg of `get_context` tool:


```python
get_context.tool_call_schema.schema()
```






### Define the graph

Let's update our ReAct agent:


```python
tools = [get_context]

# ToolNode will automatically take care of injecting Store into tools
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
# NOTE: we need to pass our store to `create_react_agent` to make sure our graph is aware of it
graph = create_react_agent(model, tools, checkpointer=checkpointer, store=doc_store)
```

### Use it!

Let's try running our graph with a `"user_id"` in the config.


```python
messages = [{"type": "user", "content": "what's the latest news about FooBar"}]
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

We can see that the tool only retrieved the correct document for user "1" when looking up the information in the store. Let's now try it again for a different user:


```python
messages = [{"type": "user", "content": "what's the latest news about FooBar"}]
config = {"configurable": {"thread_id": "2", "user_id": "2"}}
for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

We can see that the tool pulled in a different document this time.
