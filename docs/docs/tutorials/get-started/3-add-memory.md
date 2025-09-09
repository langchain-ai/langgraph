# Add memory

The chatbot can now [use tools](./2-add-tools.md) to answer user questions, but it does not remember the context of previous interactions. This limits its ability to have coherent, multi-turn conversations.

LangGraph solves this problem through **persistent checkpointing**. If you provide a `checkpointer` when compiling the graph and a `thread_id` when calling your graph, LangGraph automatically saves the state after each step. When you invoke the graph again using the same `thread_id`, the graph loads its saved state, allowing the chatbot to pick up where it left off.

We will see later that **checkpointing** is _much_ more powerful than simple chat memory - it lets you save and resume complex state at any time for error recovery, human-in-the-loop workflows, time travel interactions, and more. But first, let's add checkpointing to enable multi-turn conversations.

!!! note

    This tutorial builds on [Add tools](./2-add-tools.md).

## 1. Create a `MemorySaver` checkpointer

Create a `MemorySaver` checkpointer:

:::python

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
```

:::

:::js

```typescript
import { MemorySaver } from "@langchain/langgraph";

const memory = new MemorySaver();
```

:::

This is in-memory checkpointer, which is convenient for the tutorial. However, in a production application, you would likely change this to use `SqliteSaver` or `PostgresSaver` and connect a database.

## 2. Compile the graph

Compile the graph with the provided checkpointer, which will checkpoint the `State` as the graph works through each node:

:::python

```python
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript hl_lines="7"
const graph = new StateGraph(State)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 3. Interact with your chatbot

Now you can interact with your bot!

1.  Pick a thread to use as the key for this conversation.

    :::python

    ```python
    config = {"configurable": {"thread_id": "1"}}
    ```

    :::

    :::js

    ```typescript
    const config = { configurable: { thread_id: "1" } };
    ```

    :::

2.  Call your chatbot:

    :::python

    ```python
    user_input = "Hi there! My name is Will."

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
    ```

    ```
    ================================ Human Message =================================

    Hi there! My name is Will.
    ================================== Ai Message ==================================

    Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?
    ```

    !!! note

        The config was provided as the **second positional argument** when calling our graph. It importantly is _not_ nested within the graph inputs (`{'messages': []}`).

    :::

    :::js

    ```typescript
    const userInput = "Hi there! My name is Will.";

    const events = await graph.stream(
      { messages: [{ type: "human", content: userInput }] },
      { configurable: { thread_id: "1" }, streamMode: "values" }
    );

    for await (const event of events) {
      const lastMessage = event.messages.at(-1);
      console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
    }
    ```

    ```
    human: Hi there! My name is Will.
    ai: Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?
    ```

    !!! note
    !!! note

        The config was provided as the **second parameter** when calling our graph. It importantly is _not_ nested within the graph inputs (`{"messages": []}`).

    :::

## 4. Ask a follow up question

Ask a follow up question:

:::python

```python
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.
```

:::

:::js

```typescript
const userInput2 = "Remember my name?";

const events2 = await graph.stream(
  { messages: [{ type: "human", content: userInput2 }] },
  { configurable: { thread_id: "1" }, streamMode: "values" }
);

for await (const event of events2) {
  const lastMessage = event.messages.at(-1);
  console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
}
```

```
human: Remember my name?
ai: Yes, your name is Will. How can I help you today?
```

:::

**Notice** that we aren't using an external list for memory: it's all handled by the checkpointer! You can inspect the full execution in this [LangSmith trace](https://smith.langchain.com/public/29ba22b5-6d40-4fbe-8d27-b369e3329c84/r) to see what's going on.

Don't believe me? Try this using a different config.

:::python

```python
# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    # highlight-next-line
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

I apologize, but I don't have any previous context or memory of your name. As an AI assistant, I don't retain information from past conversations. Each interaction starts fresh. Could you please tell me your name so I can address you properly in this conversation?
```

:::

:::js

```typescript hl_lines="3-4"
const events3 = await graph.stream(
  { messages: [{ type: "human", content: userInput2 }] },
  // The only difference is we change the `thread_id` here to "2" instead of "1"
  { configurable: { thread_id: "2" }, streamMode: "values" }
);

for await (const event of events3) {
  const lastMessage = event.messages.at(-1);
  console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
}
```

```
human: Remember my name?
ai: I don't have the ability to remember personal information about users between interactions. However, I'm here to help you with any questions or topics you want to discuss!
```

:::

**Notice** that the **only** change we've made is to modify the `thread_id` in the config. See this call's [LangSmith trace](https://smith.langchain.com/public/51a62351-2f0a-4058-91cc-9996c5561428/r) for comparison.

## 5. Inspect the state

:::python

By now, we have made a few checkpoints across two different threads. But what goes into a checkpoint? To inspect a graph's `state` for a given config at any time, call `get_state(config)`.

```python
snapshot = graph.get_state(config)
snapshot
```

```
StateSnapshot(values={'messages': [HumanMessage(content='Hi there! My name is Will.', additional_kwargs={}, response_metadata={}, id='8c1ca919-c553-4ebf-95d4-b59a2d61e078'), AIMessage(content="Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?", additional_kwargs={}, response_metadata={'id': 'msg_01WTQebPhNwmMrmmWojJ9KXJ', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 405, 'output_tokens': 32}}, id='run-58587b77-8c82-41e6-8a90-d62c444a261d-0', usage_metadata={'input_tokens': 405, 'output_tokens': 32, 'total_tokens': 437}), HumanMessage(content='Remember my name?', additional_kwargs={}, response_metadata={}, id='daba7df6-ad75-4d6b-8057-745881cea1ca'), AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-93e0-6acc-8004-f2ac846575d2'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}}, 'step': 4, 'parents': {}}, created_at='2024-09-27T19:30:10.820758+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-859f-6206-8003-e1bd3c264b8f'}}, tasks=())
```

```
snapshot.next  # (since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)
```

:::

:::js

By now, we have made a few checkpoints across two different threads. But what goes into a checkpoint? To inspect a graph's `state` for a given config at any time, call `getState(config)`.

```typescript
await graph.getState({ configurable: { thread_id: "1" } });
```

```typescript
{
  values: {
    messages: [
      HumanMessage {
        "id": "32fabcef-b3b8-481f-8bcb-fd83399a5f8d",
        "content": "Hi there! My name is Will.",
        "additional_kwargs": {},
        "response_metadata": {}
      },
      AIMessage {
        "id": "chatcmpl-BrPbTsCJbVqBvXWySlYoTJvM75Kv8",
        "content": "Hello Will! How can I assist you today?",
        "additional_kwargs": {},
        "response_metadata": {},
        "tool_calls": [],
        "invalid_tool_calls": []
      },
      HumanMessage {
        "id": "561c3aad-f8fc-4fac-94a6-54269a220856",
        "content": "Remember my name?",
        "additional_kwargs": {},
        "response_metadata": {}
      },
      AIMessage {
        "id": "chatcmpl-BrPbU4BhhsUikGbW37hYuF5vvnnE2",
        "content": "Yes, I remember your name, Will! How can I help you today?",
        "additional_kwargs": {},
        "response_metadata": {},
        "tool_calls": [],
        "invalid_tool_calls": []
      }
    ]
  },
  next: [],
  tasks: [],
  metadata: {
    source: 'loop',
    step: 4,
    parents: {},
    thread_id: '1'
  },
  config: {
    configurable: {
      thread_id: '1',
      checkpoint_id: '1f05cccc-9bb6-6270-8004-1d2108bcec77',
      checkpoint_ns: ''
    }
  },
  createdAt: '2025-07-09T13:58:27.607Z',
  parentConfig: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1f05cccc-78fa-68d0-8003-ffb01a76b599'
    }
  }
}
```

```typescript
import * as assert from "node:assert";

// Since the graph ended this turn, `next` is empty.
// If you fetch a state from within a graph invocation, next tells which node will execute next)
assert.deepEqual(snapshot.next, []);
```

:::

The snapshot above contains the current state values, corresponding config, and the `next` node to process. In our case, the graph has reached an `END` state, so `next` is empty.

**Congratulations!** Your chatbot can now maintain conversation state across sessions thanks to LangGraph's checkpointing system. This opens up exciting possibilities for more natural, contextual interactions. LangGraph's checkpointing even handles **arbitrarily complex graph states**, which is much more expressive and powerful than simple chat memory.

Check out the code snippet below to review the graph from this tutorial:

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python hl_lines="36 37"
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript hl_lines="16 26"
import { END, MessagesZodState, START } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";

import { MemorySaver } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const State = z.object({
  messages: MessagesZodState.shape.messages,
});

const tools = [new TavilySearch({ maxResults: 2 })];
const llm = new ChatOpenAI({ model: "gpt-4o-mini" }).bindTools(tools);
const memory = new MemorySaver();

async function generateText(content: string) {

const graph = new StateGraph(State)
  .addNode("chatbot", async (state) => ({
    messages: [await llm.invoke(state.messages)],
  }))
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## Next steps

In the next tutorial, you will [add human-in-the-loop to the chatbot](./4-human-in-the-loop.md) to handle situations where it may need guidance or verification before proceeding.

