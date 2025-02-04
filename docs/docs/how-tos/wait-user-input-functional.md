# How to wait for user input (Functional API)

!!! info "Prerequisites"
    This guide assumes familiarity with the following:

    - Implementing [human-in-the-loop](../../concepts/human_in_the_loop.md) workflows with [interrupt](../../concepts/human_in_the_loop/#interrupt.md)
    - [How to create a ReAct agent using the Functional API](../../how-tos/react-agent-from-scratch-functional.md)

**Human-in-the-loop (HIL)** interactions are crucial for [agentic systems](../../concepts/agentic_concepts/#human-in-the-loop.md). Waiting for human input is a common HIL interaction pattern, allowing the agent to ask the user clarifying questions and await input before proceeding. 

We can implement this in LangGraph using the [interrupt()][langgraph.types.interrupt] function. `interrupt` allows us to stop graph execution to collect input from a user and continue execution with collected input.

This guide demonstrates how to implement human-in-the-loop workflows using LangGraph's [Functional API](../../concepts/functional_api.md). Specifically, we will demonstrate:

1. [A simple usage example](#simple-usage.md)
2. [How to use with a ReAct agent](#agent.md)

## Setup

First, let's install the required packages and set our API keys:


```
%%capture --no-stderr
%pip install -U langgraph langchain-openai
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
     <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for better debugging</p>
     <p style="padding-top: 5px;">
         Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM aps built with LangGraph — read more about how to get started in the <a href="https://docs.smith.langchain.com">docs</a>. 
     </p>
 </div>

## Simple usage

Let's demonstrate a simple usage example. We will create three [tasks](../../concepts/functional_api/#task.md):

1. Append `"bar"`.
2. Pause for human input. When resuming, append human input.
3. Append `"qux"`.


```python
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt


@task
def step_1(input_query):
    """Append bar."""
    return f"{input_query} bar"


@task
def human_feedback(input_query):
    """Append user input."""
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"


@task
def step_3(input_query):
    """Append qux."""
    return f"{input_query} qux"
```

We can now compose these tasks in a simple [entrypoint](../../concepts/functional_api/#entrypoint.md):


```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def graph(input_query):
    result_1 = step_1(input_query).result()
    result_2 = human_feedback(result_1).result()
    result_3 = step_3(result_2).result()

    return result_3
```

All we have done to enable human-in-the-loop workflows is called [interrupt()](../../concepts/human_in_the_loop/#interrupt.md) inside a task.

!!! tip

    The results of prior tasks-- in this case `step_1`-- are persisted, so that they are not run again following the `interrupt`.


Let's send in a query string:


```python
config = {"configurable": {"thread_id": "1"}}
```


```python
for event in graph.stream("foo", config):
    print(event)
    print("\n")
```

Note that we've paused with an `interrupt` after `step_1`. The interrupt provides instructions to resume the run. To resume, we issue a [Command](../../concepts/human_in_the_loop/#the-command-primitive.md) containing the data expected by the `human_feedback` task.


```python
# Continue execution
for event in graph.stream(Command(resume="baz"), config):
    print(event)
    print("\n")
```

After resuming, the run proceeds through the remaining step and terminates as expected.

## Agent

We will build off of the agent created in the [How to create a ReAct agent using the Functional API](../../how-tos/react-agent-from-scratch-functional.md) guide.

Here we will extend the agent by allowing it to reach out to a human for assistance when needed.

### Define model and tools

Let's first define the tools and model we will use for our example. As in the [ReAct agent guide](../../how-tos/react-agent-from-scratch-functional.md), we will use a single place-holder tool that gets a description of the weather for a location.

We will use an [OpenAI](https://python.langchain.com/docs/integrations/providers/openai/) chat model for this example, but any model [supporting tool-calling](https://python.langchain.com/docs/integrations/chat/) will suffice.


```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

model = ChatOpenAI(model="gpt-4o-mini")


@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny!"
    elif "boston" in location.lower():
        return "It's rainy!"
    else:
        return f"I am not sure what the weather is in {location}"
```

To reach out to a human for assistance, we can simply add a tool that calls [interrupt](../../concepts/human_in_the_loop/#interrupt.md):


```python
from langgraph.types import Command, interrupt


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tools = [get_weather, human_assistance]
```

### Define tasks

Our tasks are otherwise unchanged from the [ReAct agent guide](../../how-tos/react-agent-from-scratch-functional.md):

1. **Call model**: We want to query our chat model with a list of messages.
2. **Call tool**: If our model generates tool calls, we want to execute them.

We just have one more tool accessible to the model.


```python
from langchain_core.messages import ToolMessage
from langgraph.func import entrypoint, task

tools_by_name = {tool.name: tool for tool in tools}


@task
def call_model(messages):
    """Call model with a sequence of messages."""
    response = model.bind_tools(tools).invoke(messages)
    return response


@task
def call_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call)
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])
```

### Define entrypoint

Our [entrypoint](../../concepts/functional_api/#entrypoint.md) is also unchanged from the [ReAct agent guide](../../how-tos/react-agent-from-scratch-functional.md):


```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def agent(messages, previous):
    if previous is not None:
        messages = add_messages(previous, messages)

    llm_response = call_model(messages).result()
    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(messages, [llm_response, *tool_results])

        # Call model again
        llm_response = call_model(messages).result()

    # Generate final response
    messages = add_messages(messages, llm_response)
    return entrypoint.final(value=llm_response, save=messages)
```

### Usage

Let's invoke our model with a question that requires human assistance. Our question will also require an invocation of the `get_weather` tool:


```python
def _print_step(step: dict) -> None:
    for task_name, result in step.items():
        if task_name == "agent":
            continue  # just stream from tasks
        print(f"\n{task_name}:")
        if task_name == "__interrupt__":
            print(result)
        else:
            result.pretty_print()
```


```python
config = {"configurable": {"thread_id": "1"}}
```


```python
user_message = {
    "role": "user",
    "content": (
        "Can you reach out for human assistance: what should I feed my cat? "
        "Separately, can you check the weather in San Francisco?"
    ),
}
print(user_message)

for step in agent.stream([user_message], config):
    _print_step(step)
```

Note that we generate two tool calls, and although our run is interrupted, we did not block the execution of the `get_weather` tool.

Let's inspect where we're interrupted:


```python
print(step)
```

We can resume execution by issuing a [Command](../../concepts/human_in_the_loop/#the-command-primitive.md). Note that the data we supply in the `Command` can be customized to your needs based on the implementation of `human_assistance`.


```python
human_response = "You should feed your cat a fish."
human_command = Command(resume={"data": human_response})

for step in agent.stream(human_command, config):
    _print_step(step)
```

Above, when we resume we provide the final tool message, allowing the model to generate its response. Check out the LangSmith traces to see a full breakdown of the runs:

1. [Trace from initial query](https://smith.langchain.com/public/c3d8879d-4d01-41be-807e-6d9eed15df99/r)
2. [Trace after resuming](https://smith.langchain.com/public/97c05ef9-8b4c-428e-8826-3fd417c8c75f/r)
