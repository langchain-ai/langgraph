# Add tools

To handle queries that your chatbot can't answer "from memory", integrate a web search tool. The chatbot can use this tool to find relevant information and provide better responses.

!!! note

    This tutorial builds on [Build a basic chatbot](./1-build-basic-chatbot.md).

## Prerequisites

Before you start this tutorial, ensure you have the following:

:::python

- An API key for the [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/).

:::

:::js

- An API key for the [Tavily Search Engine](https://js.langchain.com/docs/integrations/tools/tavily_search/).

:::

## 1. Install the search engine

:::python
Install the requirements to use the [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/):

```bash
pip install -U langchain-tavily
```

:::

:::js
Install the requirements to use the [Tavily Search Engine](https://docs.tavily.com/):

=== "npm"

    ```bash
    npm install @langchain/community
    ```

=== "yarn"

    ```bash
    yarn add @langchain/community
    ```

=== "pnpm"

    ```bash
    pnpm add @langchain/community
    ```

=== "bun"

    ```bash
    bun add @langchain/community
    ```

:::

## 2. Configure your environment

Configure your environment with your search engine API key:

:::python

```bash
_set_env("TAVILY_API_KEY")
```

```
TAVILY_API_KEY:  ········
```

:::

:::js

```typescript
process.env.TAVILY_API_KEY = "tvly-...";
```

:::

## 3. Define the tool

Define the web search tool:

:::python

```python
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")
```

:::

:::js

```typescript
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const tool = new TavilySearchResults({ maxResults: 2 });
const tools = [tool];

await tool.invoke({ query: "What's a 'node' in LangGraph?" });
```

:::

The results are page summaries our chat bot can use to answer questions:

:::python

```
{'query': "What's a 'node' in LangGraph?",
'follow_up_questions': None,
'answer': None,
'images': [],
'results': [{'title': "Introduction to LangGraph: A Beginner's Guide - Medium",
'url': 'https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141',
'content': 'Stateful Graph: LangGraph revolves around the concept of a stateful graph, where each node in the graph represents a step in your computation, and the graph maintains a state that is passed around and updated as the computation progresses. LangGraph supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph. We define nodes for classifying the input, handling greetings, and handling search queries. def classify_input_node(state): LangGraph is a versatile tool for building complex, stateful applications with LLMs. By understanding its core concepts and working through simple examples, beginners can start to leverage its power for their projects. Remember to pay attention to state management, conditional edges, and ensuring there are no dead-end nodes in your graph.',
'score': 0.7065353,
'raw_content': None},
{'title': 'LangGraph Tutorial: What Is LangGraph and How to Use It?',
'url': 'https://www.datacamp.com/tutorial/langgraph-tutorial',
'content': 'LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner. By managing the flow of data and the sequence of operations, LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination. Whether you need a chatbot that can handle various types of user requests or a multi-agent system that performs complex tasks, LangGraph provides the tools to build exactly what you need. LangGraph significantly simplifies the development of complex LLM applications by providing a structured framework for managing state and coordinating agent interactions.',
'score': 0.5008063,
'raw_content': None}],
'response_time': 1.38}
```

:::

:::js

```
"[{\"title\":\"Introduction to LangGraph: A Beginner's Guide - Medium\",\"url\":\"https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141\",\"content\":\"Stateful Graph: LangGraph revolves around the concept of a stateful graph, where each node in the graph represents a step in your computation, and the graph maintains a state that is passed around and updated as the computation progresses. LangGraph supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph. We define nodes for classifying the input, handling greetings, and handling search queries. def classify_input_node(state): LangGraph is a versatile tool for building complex, stateful applications with LLMs. By understanding its core concepts and working through simple examples, beginners can start to leverage its power for their projects. Remember to pay attention to state management, conditional edges, and ensuring there are no dead-end nodes in your graph.\",\"score\":0.7065353,\"raw_content\":null},{\"title\":\"LangGraph Tutorial: What Is LangGraph and How to Use It?\",\"url\":\"https://www.datacamp.com/tutorial/langgraph-tutorial\",\"content\":\"LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner. By managing the flow of data and the sequence of operations, LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination. Whether you need a chatbot that can handle various types of user requests or a multi-agent system that performs complex tasks, LangGraph provides the tools to build exactly what you need. LangGraph significantly simplifies the development of complex LLM applications by providing a structured framework for managing state and coordinating agent interactions.\",\"score\":0.5008063,\"raw_content\":null}]"
```

:::

## 4. Define the graph

:::python
For the `StateGraph` you created in the [first tutorial](./1-build-basic-chatbot.md), add `bind_tools` on the LLM. This lets the LLM know the correct JSON format to use if it wants to use the search engine.
:::

:::js
For the `StateGraph` you created in the [first tutorial](./1-build-basic-chatbot.md), add `bindTools` on the LLM. This lets the LLM know the correct JSON format to use if it wants to use the search engine.
:::

Let's first select our LLM:

:::python
{!snippets/chat_model_tabs.md!}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

:::

:::js

```typescript
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });
```

:::

We can now incorporate it into a `StateGraph`:

:::python

```python hl_lines="15"
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```

:::

:::js

```typescript
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const graphBuilder = new StateGraph(StateAnnotation);

// Modification: tell the LLM which tools it can call
const llmWithTools = llm.bindTools(tools);

const chatbot = async (state: typeof StateAnnotation.State) => {
  return { messages: [await llmWithTools.invoke(state.messages)] };
};

graphBuilder.addNode("chatbot", chatbot);
```

:::

## 5. Create a function to run the tools

Now, create a function to run the tools if they are called. Do this by adding the tools to a new node called`BasicToolNode` that checks the most recent message in the state and calls tools if the message contains `tool_calls`. It relies on the LLM's `tool_calling` support, which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.

:::python

```python
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

:::

:::js

```typescript
import { ToolMessage } from "@langchain/core/messages";
import { isAIMessage } from "@langchain/core/messages";

class BasicToolNode {
  private toolsByName: Record<string, any>;

  constructor(tools: any[]) {
    this.toolsByName = {};
    for (const tool of tools) {
      this.toolsByName[tool.name] = tool;
    }
  }

  async invoke(inputs: Record<string, any>) {
    const { messages } = inputs;
    if (!messages || messages.length === 0) {
      throw new Error("No message found in input");
    }

    const message = messages[messages.length - 1];

    if (!isAIMessage(message) || !message.tool_calls) {
      throw new Error("Last message is not an AI message with tool calls");
    }

    const outputs = [];
    for (const toolCall of message.tool_calls) {
      const toolResult = await this.toolsByName[toolCall.name].invoke(
        toolCall.args
      );
      outputs.push(
        new ToolMessage({
          content: JSON.stringify(toolResult),
          name: toolCall.name,
          tool_call_id: toolCall.id,
        })
      );
    }
    return { messages: outputs };
  }
}

const toolNode = new BasicToolNode([tool]);
graphBuilder.addNode("tools", toolNode.invoke.bind(toolNode));
```

:::

!!! note

    If you do not want to build this yourself in the future, you can use LangGraph's prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode).

## 6. Define the `conditional_edges`

With the tool node added, now you can define the `conditional_edges`.

**Edges** route the control flow from one node to the next. **Conditional edges** start from a single node and usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph `state` and return a string or list of strings indicating which node(s) to call next.

:::python
Next, define a router function called `route_tools` that checks for `tool_calls` in the chatbot's output. Provide this function to the graph by calling `add_conditional_edges`, which tells the graph that whenever the `chatbot` node completes to check this function to see where to go next.
:::

:::js
Next, define a router function called `routeTools` that checks for `tool_calls` in the chatbot's output. Provide this function to the graph by calling `addConditionalEdges`, which tells the graph that whenever the `chatbot` node completes to check this function to see where to go next.
:::

The condition will route to `tools` if tool calls are present and `END` if not. Because the condition can return `END`, you do not need to explicitly set a `finish_point` this time.

:::python

```python
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

:::

:::js

```typescript
import { END, START } from "@langchain/langgraph";

const routeTools = (state: typeof StateAnnotation.State) => {
  /**
   * Use in the conditional_edge to route to the ToolNode if the last message
   * has tool calls. Otherwise, route to the end.
   */
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  if (isAIMessage(lastMessage) && lastMessage.tool_calls?.length) {
    return "tools";
  }
  return END;
};

// The `routeTools` function returns "tools" if the chatbot asks to use a tool, and "END" if
// it is fine directly responding. This conditional routing defines the main agent loop.
graphBuilder.addConditionalEdges(
  "chatbot",
  routeTools,
  // The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
  // It defaults to the identity function, but if you
  // want to use a node named something else apart from "tools",
  // You can update the value of the dictionary to something else
  // e.g., "tools": "my_tools"
  { tools: "tools", [END]: END }
);

// Any time a tool is called, we return to the chatbot to decide the next step
graphBuilder.addEdge("tools", "chatbot");
graphBuilder.addEdge(START, "chatbot");

const graph = graphBuilder.compile();
```

:::

!!! note

    You can replace this with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition) to be more concise.

## 7. Visualize the graph (optional)

:::python
You can visualize the graph using the `get_graph` method and one of the "draw" methods, like `draw_ascii` or `draw_png`. The `draw` methods each require additional dependencies.

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![chatbot-with-tools-diagram](chatbot-with-tools.png)
:::

:::js
You can visualize the graph using the `getGraph` method and one of the "draw" methods, like `drawAscii` or `drawMermaidPng`. The `draw` methods each require additional dependencies.

```typescript
import * as tslab from "tslab";

try {
  const graphRepresentation = graph.getGraph();
  const image = await graphRepresentation.drawMermaidPng();
  const arrayBuffer = await image.arrayBuffer();
  await tslab.display.png(new Uint8Array(arrayBuffer));
} catch (error) {
  // This requires some extra dependencies and is optional
  console.log("Could not render graph");
}
```

:::

## 8. Ask the bot questions

Now you can ask the chatbot questions outside its training data:

:::python

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

```
Assistant: [{'text': "To provide you with accurate and up-to-date information about LangGraph, I'll need to search for the latest details. Let me do that for you.", 'type': 'text'}, {'id': 'toolu_01Q588CszHaSvvP2MxRq9zRD', 'input': {'query': 'LangGraph AI tool information'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Assistant: [{"url": "https://www.langchain.com/langgraph", "content": "LangGraph sets the foundation for how we can build and scale AI workloads \u2014 from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "Overview. LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures ..."}]
Assistant: Based on the search results, I can provide you with information about LangGraph:

1. Purpose:
   LangGraph is a library designed for building stateful, multi-actor applications with Large Language Models (LLMs). It's particularly useful for creating agent and multi-agent workflows.

2. Developer:
   LangGraph is developed by LangChain, a company known for its tools and frameworks in the AI and LLM space.

3. Key Features:
   - Cycles: LangGraph allows the definition of flows that involve cycles, which is essential for most agentic architectures.
   - Controllability: It offers enhanced control over the application flow.
   - Persistence: The library provides ways to maintain state and persistence in LLM-based applications.

4. Use Cases:
   LangGraph can be used for various applications, including:
   - Conversational agents
   - Complex task automation
   - Custom LLM-backed experiences

5. Integration:
   LangGraph works in conjunction with LangSmith, another tool by LangChain, to provide an out-of-the-box solution for building complex, production-ready features with LLMs.

6. Significance:
...
   LangGraph is noted to offer unique benefits compared to other LLM frameworks, particularly in its ability to handle cycles, provide controllability, and maintain persistence.

LangGraph appears to be a significant tool in the evolving landscape of LLM-based application development, offering developers new ways to create more complex, stateful, and interactive AI systems.
Goodbye!
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

:::

:::js

```typescript
import { HumanMessage } from "@langchain/core/messages";

const streamGraphUpdates = async (userInput: string) => {
  const stream = await graph.stream(
    { messages: [new HumanMessage(userInput)] },
    { streamMode: "values" }
  );

  for await (const event of stream) {
    const lastMessage = event.messages[event.messages.length - 1];
    console.log("Assistant:", lastMessage.content);
  }
};

// Example usage
const userInput = "What do you know about LangGraph?";
console.log("User:", userInput);
await streamGraphUpdates(userInput);
```

```
User: What do you know about LangGraph?
Assistant: I'll search for the latest information about LangGraph for you.
Assistant: [{"title":"Introduction to LangGraph: A Beginner's Guide - Medium","url":"https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141","content":"..."}]
Assistant: Based on the search results, I can provide you with information about LangGraph:

LangGraph is a library within the LangChain ecosystem designed for building stateful, multi-actor applications with Large Language Models (LLMs). Here are the key aspects:

**Core Purpose:**
- LangGraph is specifically designed for creating agent and multi-agent workflows
- It provides a framework for defining, coordinating, and executing multiple LLM agents in a structured manner

**Key Features:**
1. **Stateful Graph Architecture**: LangGraph revolves around a stateful graph where each node represents a step in computation, and the graph maintains state that is passed around and updated as the computation progresses

2. **Conditional Edges**: It supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph

3. **Cycles**: Unlike other LLM frameworks, LangGraph allows you to define flows that involve cycles, which is essential for most agentic architectures

4. **Controllability**: It offers enhanced control over the application flow

5. **Persistence**: The library provides ways to maintain state and persistence in LLM-based applications

**Use Cases:**
- Conversational agents
- Complex task automation
- Custom LLM-backed experiences
- Multi-agent systems that perform complex tasks

**Benefits:**
LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination, making it easier to build complex, production-ready features with LLMs.

This makes LangGraph a significant tool in the evolving landscape of LLM-based application development.
```

:::

## 9. Use prebuilts

For ease of use, adjust your code to replace the following with LangGraph prebuilt components. These have built in functionality like parallel API execution.

:::python

- `BasicToolNode` is replaced with the prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
- `route_tools` is replaced with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)

{!snippets/chat_model_tabs.md!}

```python hl_lines="25 30"
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

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
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

:::

:::js

- `BasicToolNode` is replaced with the prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
- `routeTools` is replaced with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)

```typescript
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { StateGraph, START } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

const graphBuilder = new StateGraph(StateAnnotation);

const tool = new TavilySearchResults({ maxResults: 2 });
const tools = [tool];

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });
const llmWithTools = llm.bindTools(tools);

const chatbot = async (state: typeof StateAnnotation.State) => {
  return { messages: [await llmWithTools.invoke(state.messages)] };
};

graphBuilder.addNode("chatbot", chatbot);

const toolNode = new ToolNode(tools);
graphBuilder.addNode("tools", toolNode);

graphBuilder.addConditionalEdges("chatbot", toolsCondition);
// Any time a tool is called, we return to the chatbot to decide the next step
graphBuilder.addEdge("tools", "chatbot");
graphBuilder.addEdge(START, "chatbot");

const graph = graphBuilder.compile();
```

:::

**Congratulations!** You've created a conversational agent in LangGraph that can use a search engine to retrieve updated information when needed. Now it can handle a wider range of user queries. :::python To inspect all the steps your agent just took, check out this [LangSmith trace](https://smith.langchain.com/public/4fbd7636-25af-4638-9587-5a02fdbb0172/r). :::

## Next steps

The chatbot cannot remember past interactions on its own, which limits its ability to have coherent, multi-turn conversations. In the next part, you will [add **memory**](./3-add-memory.md) to address this.
