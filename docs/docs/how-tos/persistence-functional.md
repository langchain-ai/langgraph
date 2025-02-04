# How to add thread-level persistence (functional API)

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Functional API](../../concepts/functional_api/.md)
    - [Persistence](../../concepts/persistence/.md)
    - [Memory](../../concepts/memory/.md)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

Many AI applications need memory to share context across multiple interactions on the same [thread](../../concepts/persistence#threads.md) (e.g., multiple turns of a conversation). In LangGraph functional API, this kind of memory can be added to any [entrypoint()][langgraph.func.entrypoint] workflow using [thread-level persistence](https://langchain-ai.github.io/langgraph/concepts/persistence).

When creating a LangGraph workflow, you can set it up to persist its results by using a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#basecheckpointsaver):


1. Create an instance of a checkpointer:

    ```python
    from langgraph.checkpoint.memory import MemorySaver
    
    checkpointer = MemorySaver()       
    ```

2. Pass `checkpointer` instance to the `entrypoint()` decorator:

    ```python
    from langgraph.func import entrypoint
    
    @entrypoint(checkpointer=checkpointer)
    def workflow(inputs)
        ...
    ```

3. Optionally expose `previous` parameter in the workflow function signature:

    ```python
    @entrypoint(checkpointer=checkpointer)
    def workflow(
        inputs,
        *,
        # you can optionally specify `previous` in the workflow function signature
        # to access the return value from the workflow as of the last execution
        previous
    ):
        previous = previous or []
        combined_inputs = previous + inputs
        result = do_something(combined_inputs)
        ...
    ```

4. Optionally choose which values will be returned from the workflow and which will be saved by the checkpointer as `previous`:

    ```python
    @entrypoint(checkpointer=checkpointer)
    def workflow(inputs, *, previous):
        ...
        result = do_something(...)
        return entrypoint.final(value=result, save=combine(inputs, result))
    ```

This guide shows how you can add thread-level persistence to your workflow.

!!! tip "Note"

    If you need memory that is __shared__ across multiple conversations or users (cross-thread persistence), check out this [how-to guide](../cross-thread-persistence-functional.md).

!!! tip "Note"

    If you need to add thread-level persistence to a `StateGraph`, check out this [how-to guide](../persistence.md).

## Setup

First we need to install the packages required


```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

Next, we need to set API key for Anthropic (the LLM we will use).


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Example: simple chatbot with short-term memory

We will be using a workflow with a single task that calls a [chat model](https://python.langchain.com/docs/concepts/chat_models/).

Let's first define the model we'll be using:


```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-latest")
```

Now we can define our task and workflow. To add in persistence, we need to pass in a [Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) to the [entrypoint()][langgraph.func.entrypoint] decorator.


```python
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver


@task
def call_model(messages: list[BaseMessage]):
    response = model.invoke(messages)
    return response


checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def workflow(inputs: list[BaseMessage], *, previous: list[BaseMessage]):
    if previous:
        inputs = add_messages(previous, inputs)

    response = call_model(inputs).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))
```

If we try to use this workflow, the context of the conversation will be persisted across interactions:

!!! note Note

    If you're using LangGraph Cloud or LangGraph Studio, you __don't need__ to pass checkpointer to the entrypoint decorator, since it's done automatically.

We can now interact with the agent and see that it remembers previous messages!


```python
config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

You can always resume previous threads:


```python
input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

If we want to start a new conversation, we can pass in a different `thread_id`. Poof! All the memories are gone!


```python
input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream(
    [input_message],
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk.pretty_print()
```

!!! tip "Streaming tokens"

    If you would like to stream LLM tokens from your chatbot, you can use `stream_mode="messages"`. Check out this [how-to guide](../streaming-tokens.md) to learn more.
