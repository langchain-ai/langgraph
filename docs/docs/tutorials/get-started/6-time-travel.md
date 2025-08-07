# Time travel

In a typical chatbot workflow, the user interacts with the bot one or more times to accomplish a task. [Memory](./3-add-memory.md) and a [human-in-the-loop](./4-human-in-the-loop.md) enable checkpoints in the graph state and control future responses.

What if you want a user to be able to start from a previous response and explore a different outcome? Or what if you want users to be able to rewind your chatbot's work to fix mistakes or try a different strategy, something that is common in applications like autonomous software engineers?

You can create these types of experiences using LangGraph's built-in **time travel** functionality.

!!! note

    This tutorial builds on [Customize state](./5-customize-state.md).

## 1. Rewind your graph

:::python
Rewind your graph by fetching a checkpoint using the graph's `get_state_history` method. You can then resume execution at this previous point in time.
:::

:::js
Rewind your graph by fetching a checkpoint using the graph's `getStateHistory` method. You can then resume execution at this previous point in time.
:::

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
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
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript
import {
  StateGraph,
  START,
  END,
  MessagesZodState,
  MemorySaver,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const tools = [new TavilySearch({ maxResults: 2 })];
const llmWithTools = new ChatOpenAI({ model: "gpt-4o-mini" }).bindTools(tools);
const memory = new MemorySaver();

const graph = new StateGraph(State)
  .addNode("chatbot", async (state) => ({
    messages: [await llmWithTools.invoke(state.messages)],
  }))
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 2. Add steps

Add steps to your graph. Every step will be checkpointed in its state history:

:::python

```python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

I'm learning LangGraph. Could you do some research on it for me?
================================== Ai Message ==================================

[{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01BscbfJJB9EWJFqGrN6E54e', 'input': {'query': 'LangGraph latest information and features'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01BscbfJJB9EWJFqGrN6E54e)
 Call ID: toolu_01BscbfJJB9EWJFqGrN6E54e
  Args:
    query: LangGraph latest information and features
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://blockchain.news/news/langchain-new-features-upcoming-events-update", "content": "LangChain, a leading platform in the AI development space, has released its latest updates, showcasing new use cases and enhancements across its ecosystem. According to the LangChain Blog, the updates cover advancements in LangGraph Platform, LangSmith's self-improving evaluators, and revamped documentation for LangGraph."}, {"url": "https://blog.langchain.dev/langgraph-platform-announce/", "content": "With these learnings under our belt, we decided to couple some of our latest offerings under LangGraph Platform. LangGraph Platform today includes LangGraph Server, LangGraph Studio, plus the CLI and SDK. ... we added features in LangGraph Server to deliver on a few key value areas. Below, we'll focus on these aspects of LangGraph Platform."}]
================================== Ai Message ==================================

Thank you for your patience. I've found some recent information about LangGraph for you. Let me summarize the key points:

1. LangGraph is part of the LangChain ecosystem, which is a leading platform in AI development.

2. Recent updates and features of LangGraph include:

   a. LangGraph Platform: This seems to be a cloud-based version of LangGraph, though specific details weren't provided in the search results.
...
3. Keep an eye on LangGraph Platform developments, as cloud-based solutions often provide an easier starting point for learners.
4. Consider how LangGraph fits into the broader LangChain ecosystem, especially its interaction with tools like LangSmith.

Is there any specific aspect of LangGraph you'd like to know more about? I'd be happy to do a more focused search on particular features or use cases.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

```python
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...
6. Pay attention to how you structure the agent's decision-making process and workflow.
7. Don't forget to implement proper error handling and safety measures, especially if your agent will be interacting with external systems or making important decisions.

Building an autonomous agent is an iterative process, so be prepared to refine and improve your agent over time. Good luck with your project! If you need any more specific information as you progress, feel free to ask.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

:::

:::js

```typescript
import { randomUUID } from "node:crypto";
const threadId = randomUUID();

let iter = 0;

for (const userInput of [
  "I'm learning LangGraph. Could you do some research on it for me?",
  "Ya that's helpful. Maybe I'll build an autonomous agent with it!",
]) {
  iter += 1;

  console.log(`\n--- Conversation Turn ${iter} ---\n`);
  const events = await graph.stream(
    { messages: [{ role: "user", content: userInput }] },
    { configurable: { thread_id: threadId }, streamMode: "values" }
  );

  for await (const event of events) {
    if ("messages" in event) {
      const lastMessage = event.messages.at(-1);

      console.log(
        "=".repeat(32),
        `${lastMessage?.getType()} Message`,
        "=".repeat(32)
      );
      console.log(lastMessage?.text);
    }
  }
}
```

```
--- Conversation Turn 1 ---

================================ human Message ================================
I'm learning LangGraph.js. Could you do some research on it for me?
================================ ai Message ================================
I'll search for information about LangGraph.js for you.
================================ tool Message ================================
{
  "query": "LangGraph.js framework TypeScript langchain what is it tutorial guide",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/an-absolute-beginners-guide-to-langgraph-js/4212496",
      "title": "An Absolute Beginner's Guide to LangGraph.js",
      "content": "(...)",
      "score": 0.79369855,
      "raw_content": null
    },
    {
      "url": "https://langchain-ai.github.io/langgraphjs/",
      "title": "LangGraph.js",
      "content": "(...)",
      "score": 0.78154784,
      "raw_content": null
    }
  ],
  "response_time": 2.37
}
================================ ai Message ================================
Let me provide you with an overview of LangGraph.js based on the search results:

LangGraph.js is a JavaScript/TypeScript library that's part of the LangChain ecosystem, specifically designed for creating and managing complex LLM (Large Language Model) based workflows. Here are the key points about LangGraph.js:

1. Purpose:
- It's a low-level orchestration framework for building controllable agents
- Particularly useful for creating agentic workflows where LLMs decide the course of action based on current state
- Helps model workflows as graphs with nodes and edges

(...)

--- Conversation Turn 2 ---

================================ human Message ================================
Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================ ai Message ================================
Let me search for specific information about building autonomous agents with LangGraph.js.
================================ tool Message ================================
{
  "query": "how to build autonomous agents with LangGraph.js examples tutorial react agent",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://ai.google.dev/gemini-api/docs/langgraph-example",
      "title": "ReAct agent from scratch with Gemini 2.5 and LangGraph",
      "content": "(...)",
      "score": 0.7602419,
      "raw_content": null
    },
    {
      "url": "https://www.youtube.com/watch?v=ZfjaIshGkmk",
      "title": "Build Autonomous AI Agents with ReAct and LangGraph Tools",
      "content": "(...)",
      "score": 0.7471924,
      "raw_content": null
    }
  ],
  "response_time": 1.98
}
================================ ai Message ================================
Based on the search results, I can provide you with a practical overview of how to build an autonomous agent with LangGraph.js. Here's what you need to know:

1. Basic Structure for Building an Agent:
- LangGraph.js provides a ReAct (Reason + Act) pattern implementation
- The basic components include:
  - State management for conversation history
  - Nodes for different actions
  - Edges for decision-making flow
  - Tools for specific functionalities

(...)

```

:::

## 3. Replay the full state history

Now that you have added steps to the chatbot, you can `replay` the full state history to see everything that occurred.

:::python

```python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
```

```
Num Messages:  8 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  7 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  6 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  5 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ('__start__',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  3 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  2 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  1 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  0 Next:  ('__start__',)
--------------------------------------------------------------------------------
```

:::

:::js

```typescript
import type { StateSnapshot } from "@langchain/langgraph";

let toReplay: StateSnapshot | undefined;
for await (const state of graph.getStateHistory({
  configurable: { thread_id: threadId },
})) {
  console.log(
    `Num Messages: ${state.values.messages.length}, Next: ${JSON.stringify(
      state.next
    )}`
  );
  console.log("-".repeat(80));
  if (state.values.messages.length === 6) {
    // We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
    toReplay = state;
  }
}
```

```
Num Messages: 8 Next:  []
--------------------------------------------------------------------------------
Num Messages: 7 Next:  ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 6 Next:  ["tools"]
--------------------------------------------------------------------------------
Num Messages: 7, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 6, Next: ["tools"]
--------------------------------------------------------------------------------
Num Messages: 5, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 4, Next: ["__start__"]
--------------------------------------------------------------------------------
Num Messages: 4, Next: []
--------------------------------------------------------------------------------
Num Messages: 3, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 2, Next: ["tools"]
--------------------------------------------------------------------------------
Num Messages: 1, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 0, Next: ["__start__"]
--------------------------------------------------------------------------------
```

:::

Checkpoints are saved for every step of the graph. This **spans invocations** so you can rewind across a full thread's history.

## Resume from a checkpoint

:::python

Resume from the `to_replay` state, which is after the `chatbot` node in the second graph invocation. Resuming from this point will call the **action** node next.
:::

:::js
Resume from the `toReplay` state, which is after a specific node in one of the graph invocations. Resuming from this point will call the next scheduled node.
:::

:::python

```python
print(to_replay.next)
print(to_replay.config)
```

```
('tools',)
{'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1efd43e3-0c1f-6c4e-8006-891877d65740'}}
```

:::

:::js

Resume from the `toReplay` state, which is after the `chatbot` node in one of the graph invocations. Resuming from this point will call the next scheduled node.

```typescript
console.log(toReplay.next);
console.log(toReplay.config);
```

```
["tools"]
{
  configurable: {
    thread_id: "007708b8-ea9b-4ff7-a7ad-3843364dbf75",
    checkpoint_ns: "",
    checkpoint_id: "1efd43e3-0c1f-6c4e-8006-891877d65740"
  }
}
```

:::

## 4. Load a state from a moment-in-time

:::python

The checkpoint's `to_replay.config` contains a `checkpoint_id` timestamp. Providing this `checkpoint_id` value tells LangGraph's checkpointer to **load** the state from that moment in time.

```python
# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...

Remember, building an autonomous agent is an iterative process. Start simple and gradually increase complexity as you become more comfortable with LangGraph and its capabilities.

Would you like more information on any specific aspect of building your autonomous agent with LangGraph?
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

The graph resumed execution from the `tools` node. You can tell this is the case since the first value printed above is the response from our search engine tool.
:::

:::js

The checkpoint's `toReplay.config` contains a `checkpoint_id` timestamp. Providing this `checkpoint_id` value tells LangGraph's checkpointer to **load** the state from that moment in time.

```typescript
// The `checkpoint_id` in the `toReplay.config` corresponds to a state we've persisted to our checkpointer.
for await (const event of await graph.stream(null, {
  ...toReplay?.config,
  streamMode: "values",
})) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);

    console.log(
      "=".repeat(32),
      `${lastMessage?.getType()} Message`,
      "=".repeat(32)
    );
    console.log(lastMessage?.text);
  }
}
```

```
================================ ai Message ================================
Let me search for specific information about building autonomous agents with LangGraph.js.
================================ tool Message ================================
{
  "query": "how to build autonomous agents with LangGraph.js examples tutorial",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://www.mongodb.com/developer/languages/typescript/build-javascript-ai-agent-langgraphjs-mongodb/",
      "title": "Build a JavaScript AI Agent With LangGraph.js and MongoDB",
      "content": "(...)",
      "score": 0.7672197,
      "raw_content": null
    },
    {
      "url": "https://medium.com/@lorevanoudenhove/how-to-build-ai-agents-with-langgraph-a-step-by-step-guide-5d84d9c7e832",
      "title": "How to Build AI Agents with LangGraph: A Step-by-Step Guide",
      "content": "(...)",
      "score": 0.7407191,
      "raw_content": null
    }
  ],
  "response_time": 0.82
}
================================ ai Message ================================
Based on the search results, I can share some practical information about building autonomous agents with LangGraph.js. Here are some concrete examples and approaches:

1. Example HR Assistant Agent:
- Can handle HR-related queries using employee information
- Features include:
  - Starting and continuing conversations
  - Looking up information using vector search
  - Persisting conversation state using checkpoints
  - Managing threaded conversations

2. Energy Savings Calculator Agent:
- Functions as a lead generation tool for solar panel sales
- Capabilities include:
  - Calculating potential energy savings
  - Handling multi-step conversations
  - Processing user inputs for personalized estimates
  - Managing conversation state

(...)
```

The graph resumed execution from the `tools` node. You can tell this is the case since the first value printed above is the response from our search engine tool.
:::

**Congratulations!** You've now used time-travel checkpoint traversal in LangGraph. Being able to rewind and explore alternative paths opens up a world of possibilities for debugging, experimentation, and interactive applications.

## Learn more

Take your LangGraph journey further by exploring deployment and advanced features:

- **[LangGraph Server quickstart](../../tutorials/langgraph-platform/local-server.md)**: Launch a LangGraph server locally and interact with it using the REST API and LangGraph Studio Web UI.
- **[LangGraph Platform quickstart](../../cloud/quick_start.md)**: Deploy your LangGraph app using LangGraph Platform.
- **[LangGraph Platform concepts](../../concepts/langgraph_platform.md)**: Understand the foundational concepts of the LangGraph Platform.
