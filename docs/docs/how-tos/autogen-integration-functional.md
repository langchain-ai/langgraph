# How to integrate LangGraph (functional API) with AutoGen, CrewAI, and other frameworks

LangGraph is a framework for building agentic and multi-agent applications. LangGraph can be easily integrated with other agent frameworks. 

The primary reasons you might want to integrate LangGraph with other agent frameworks:

- create [multi-agent systems](../../concepts/multi_agent.md) where individual agents are built with different frameworks
- leverage LangGraph to add features like [persistence](../../concepts/persistence.md), [streaming](../../concepts/streaming.md), [short and long-term memory](../../concepts/memory.md) and more

The simplest way to integrate agents from other frameworks is by calling those agents inside a LangGraph [node](../../concepts/low_level/#nodes.md):

```python
import autogen
from langgraph.func import entrypoint, task

autogen_agent = autogen.AssistantAgent(name="assistant", ...)
user_proxy = autogen.UserProxyAgent(name="user_proxy", ...)

@task
def call_autogen_agent(messages):
    response = user_proxy.initiate_chat(
        autogen_agent,
        message=messages[-1],
        ...
    )
    ...


@entrypoint()
def workflow(messages):
    response = call_autogen_agent(messages).result()
    return response


workflow.invoke(
    [
        {
            "role": "user",
            "content": "Find numbers between 10 and 30 in fibonacci sequence",
        }
    ]
)
```

In this guide we show how to build a LangGraph chatbot that integrates with AutoGen, but you can follow the same approach with other frameworks.

## Setup


```
%pip install autogen langgraph
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## Define AutoGen agent

Here we define our AutoGen agent. Adapted from official tutorial [here](https://github.com/microsoft/autogen/blob/0.2/notebook/agentchat_web_info.ipynb).


```python
import autogen
import os

config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

autogen_agent = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
)
```

---

## Create the workflow

We will now create a LangGraph chatbot graph that calls AutoGen agent.


```python
from typing import Literal, TypedDict

from langchain_core.messages import convert_to_openai_messages, BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver


@task
def call_autogen_agent(messages: list[BaseMessage]):
    # convert to openai-style messages
    messages = convert_to_openai_messages(messages)
    response = user_proxy.initiate_chat(
        autogen_agent,
        message=messages[-1],
        # pass previous message history as context
        carryover=messages[:-1],
    )
    # get the final response from the agent
    content = response.chat_history[-1]["content"]
    return {"role": "assistant", "content": content}


# add short-term memory for storing conversation history
checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def workflow(messages: list[BaseMessage], previous: list[BaseMessage]):
    messages = add_messages(previous or [], messages)
    response = call_autogen_agent(messages).result()
    return entrypoint.final(value=response, save=add_messages(messages, response))
```

## Run the graph

We can now run the graph.


```python
# pass the thread ID to persist agent outputs for future interactions
# highlight-next-line
config = {"configurable": {"thread_id": "1"}}

for chunk in workflow.stream(
    [
        {
            "role": "user",
            "content": "Find numbers between 10 and 30 in fibonacci sequence",
        }
    ],
    # highlight-next-line
    config,
):
    print(chunk)
```

Since we're leveraging LangGraph's [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) features we can now continue the conversation using the same thread ID -- LangGraph will automatically pass previous history to the AutoGen agent:


```python
for chunk in workflow.stream(
    [
        {
            "role": "user",
            "content": "Multiply the last number by 3",
        }
    ],
    # highlight-next-line
    config,
):
    print(chunk)
```
