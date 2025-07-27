---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Tools

:::python
[Tools](https://python.langchain.com/docs/concepts/tools/) are a way to encapsulate a function and its input schema in a way that can be passed to a chat model that supports tool calling. This allows the model to request the execution of this function with specific inputs.

You can either [define your own tools](#define-simple-tools) or use [prebuilt integrations](#prebuilt-tools) that LangChain provides.

## Define simple tools

You can pass a vanilla function to `create_react_agent` to use as a tool:

```python
from langgraph.prebuilt import create_react_agent

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

create_react_agent(
    model="anthropic:claude-3-7-sonnet",
    tools=[multiply]
)
```

`create_react_agent` automatically converts vanilla functions to [LangChain tools](https://python.langchain.com/docs/concepts/tools/#tool-interface).
:::

:::js
[Tools](https://js.langchain.com/docs/concepts/tools/) are a way to encapsulate a function and its input schema in a way that can be passed to a chat model that supports tool calling. This allows the model to request the execution of this function with specific inputs.

You can either [define your own tools](#define-simple-tools) or use [prebuilt integrations](#prebuilt-tools) that LangChain provides.

## Define simple tools

You can pass a vanilla function to `createReactAgent` to use as a tool:

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

function multiply(a: number, b: number): number {
  return a * b;
}

createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-7-sonnet" }),
  tools: [multiply],
});
```

`createReactAgent` automatically converts vanilla functions to [LangChain tools](https://js.langchain.com/docs/concepts/tools/#tool-interface).
:::

## Customize tools

:::python
For more control over tool behavior, use the `@tool` decorator:

```python
# highlight-next-line
from langchain_core.tools import tool

# highlight-next-line
@tool("multiply_tool", parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b
```

You can also define a custom input schema using Pydantic:

```python
from pydantic import BaseModel, Field

class MultiplyInputSchema(BaseModel):
    """Multiply two numbers"""
    a: int = Field(description="First operand")
    b: int = Field(description="Second operand")

# highlight-next-line
@tool("multiply_tool", args_schema=MultiplyInputSchema)
def multiply(a: int, b: int) -> int:
    return a * b
```

For additional customization, refer to the [custom tools guide](https://python.langchain.com/docs/how_to/custom_tools/).
:::

:::js
For more control over tool behavior, use the `tool` function:

```typescript
// highlight-next-line
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// highlight-next-line
const multiply = tool(
  (input) => {
    return input.a * input.b;
  },
  {
    name: "multiply_tool",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number().describe("First operand"),
      b: z.number().describe("Second operand"),
    }),
  }
);
```

For additional customization, refer to the [custom tools guide](https://js.langchain.com/docs/how_to/custom_tools/).
:::

## Hide arguments from the model

Some tools require runtime-only arguments (e.g., user ID or session context) that should not be controllable by the model.

You can put these arguments in the `state` or `config` of the agent, and access
this information inside the tool:

:::python

```python
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

def my_tool(
    # This will be populated by an LLM
    tool_arg: str,
    # access information that's dynamically updated inside the agent
    # highlight-next-line
    state: Annotated[AgentState, InjectedState],
    # access static data that is passed at agent invocation
    # highlight-next-line
    config: RunnableConfig,
) -> str:
    """My tool."""
    do_something_with_state(state["messages"])
    do_something_with_config(config)
    ...
```

:::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import { z } from "zod";

const myTool = tool(
  async (input, config: LangGraphRunnableConfig) => {
    // This will be populated by an LLM
    const toolArg = input.toolArg;

    // access information that's dynamically updated inside the agent
    // highlight-next-line
    const state = config.store;

    // access static data that is passed at agent invocation
    // highlight-next-line
    const userId = config.configurable?.userId;

    // Use state and config in your tool logic
    return "Tool result";
  },
  {
    name: "my_tool",
    description: "My tool",
    schema: z.object({
      toolArg: z.string().describe("Tool argument"),
    }),
  }
);
```

:::

## Disable parallel tool calling

Some model providers support executing multiple tools in parallel, but
allow users to disable this feature.

:::python
For supported providers, you can disable parallel tool calling by setting `parallel_tool_calls=False` via the `model.bind_tools()` method:

```python
from langchain.chat_models import init_chat_model

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0)
tools = [add, multiply]
agent = create_react_agent(
    # disable parallel tool calls
    # highlight-next-line
    model=model.bind_tools(tools, parallel_tool_calls=False),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what's 3 + 5 and 4 * 7?"}]}
)
```

:::

:::js
For supported providers, you can disable parallel tool calling by setting `parallel_tool_calls: false` via the `bindTools()` method:

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const add = tool((input) => input.a + input.b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number(),
    b: z.number(),
  }),
});

const multiply = tool((input) => input.a * input.b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number(),
    b: z.number(),
  }),
});

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
  temperature: 0,
});
const tools = [add, multiply];

const agent = createReactAgent({
  // disable parallel tool calls
  // highlight-next-line
  llm: model.bindTools(tools, { parallel_tool_calls: false }),
  tools,
});

await agent.invoke({
  messages: [{ role: "user", content: "what's 3 + 5 and 4 * 7?" }],
});
```

:::

## Return tool results directly

:::python
Use `return_direct=True` to return tool results immediately and stop the agent loop:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool(return_direct=True)
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[add]
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what's 3 + 5?"}]}
)
```

:::

:::js
Use `returnDirect: true` to return tool results immediately and stop the agent loop:

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// highlight-next-line
const add = tool((input) => input.a + input.b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number(),
    b: z.number(),
  }),
  // highlight-next-line
  returnDirect: true,
});

const agent = createReactAgent({
  llm: model,
  tools: [add],
});

await agent.invoke({
  messages: [{ role: "user", content: "what's 3 + 5?" }],
});
```

:::

## Force tool use

:::python
To force the agent to use specific tools, you can set the `tool_choice` option in `model.bind_tools()`:

```python
from langchain_core.tools import tool

# highlight-next-line
@tool(return_direct=True)
def greet(user_name: str) -> int:
    """Greet user."""
    return f"Hello {user_name}!"

tools = [greet]

agent = create_react_agent(
    # highlight-next-line
    model=model.bind_tools(tools, tool_choice={"type": "tool", "name": "greet"}),
    tools=tools
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, I am Bob"}]}
)
```

:::

:::js
To force the agent to use specific tools, you can set the `tool_choice` option in `bindTools()`:

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// highlight-next-line
const greet = tool((input) => `Hello ${input.userName}!`, {
  name: "greet",
  description: "Greet user",
  schema: z.object({
    userName: z.string(),
  }),
  // highlight-next-line
  returnDirect: true,
});

const tools = [greet];

const agent = createReactAgent({
  // highlight-next-line
  llm: model.bindTools(tools, { tool_choice: { type: "tool", name: "greet" } }),
  tools,
});

await agent.invoke({
  messages: [{ role: "user", content: "Hi, I am Bob" }],
});
```

:::

!!! Warning "Avoid infinite loops"

    Forcing tool usage without stopping conditions can create infinite loops. Use one of the following safeguards:

    - Mark the tool with [`return_direct=True`](#return-tool-results-directly) to end the loop after execution.
    - Set [`recursion_limit`](../concepts/low_level.md#recursion-limit) to restrict the number of execution steps.

## Handle tool errors

:::python
By default, the agent will catch all exceptions raised during tool calls and will pass those as tool messages to the LLM. To control how the errors are handled, you can use the prebuilt [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] — the node that executes tools inside `create_react_agent` — via its `handle_tool_errors` parameter:
:::

:::js
By default, the agent will catch all exceptions raised during tool calls and will pass those as tool messages to the LLM. To control how the errors are handled, you can use the prebuilt [`ToolNode`][<insert-ref>] — the node that executes tools inside `createReactAgent` — via its `handleToolErrors` parameter:
:::

=== "Enable error handling (default)"

    :::python
    ```python
    from langgraph.prebuilt import create_react_agent

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # Run with error handling (default)
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[multiply]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```
    :::

    :::js
    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const multiply = tool(
      (input) => {
        if (input.a === 42) {
          throw new Error("The ultimate error");
        }
        return input.a * input.b;
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
      }
    );

    // Run with error handling (default)
    const agent = createReactAgent({
      llm: new ChatAnthropic({ model: "claude-3-7-sonnet-latest" }),
      tools: [multiply]
    });

    await agent.invoke({
      messages: [{ role: "user", content: "what's 42 x 7?" }]
    });
    ```
    :::

=== "Disable error handling"

    :::python
    ```python
    from langgraph.prebuilt import create_react_agent, ToolNode

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # highlight-next-line
    tool_node = ToolNode(
        [multiply],
        # highlight-next-line
        handle_tool_errors=False  # (1)!
    )
    agent_no_error_handling = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=tool_node
    )
    agent_no_error_handling.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```

    1. This disables error handling (enabled by default). See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].
    :::

    :::js
    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { ToolNode } from "@langchain/langgraph/prebuilt";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const multiply = tool(
      (input) => {
        if (input.a === 42) {
          throw new Error("The ultimate error");
        }
        return input.a * input.b;
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
      }
    );

    // highlight-next-line
    const toolNode = new ToolNode(
      [multiply],
      {
        // highlight-next-line
        handleToolErrors: false  // (1)!
      }
    );

    const agentNoErrorHandling = createReactAgent({
      llm: new ChatAnthropic({ model: "claude-3-7-sonnet-latest" }),
      tools: toolNode
    });

    await agentNoErrorHandling.invoke({
      messages: [{ role: "user", content: "what's 42 x 7?" }]
    });
    ```

    1. This disables error handling (enabled by default). See all available strategies in the [API reference][toolnode].
    :::

=== "Custom error handling"

    :::python
    ```python
    from langgraph.prebuilt import create_react_agent, ToolNode

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a == 42:
            raise ValueError("The ultimate error")
        return a * b

    # highlight-next-line
    tool_node = ToolNode(
        [multiply],
        # highlight-next-line
        handle_tool_errors=(
            "Can't use 42 as a first operand, you must switch operands!"  # (1)!
        )
    )
    agent_custom_error_handling = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=tool_node
    )
    agent_custom_error_handling.invoke(
        {"messages": [{"role": "user", "content": "what's 42 x 7?"}]}
    )
    ```

    1. This provides a custom message to send to the LLM in case of an exception. See all available strategies in the [API reference][langgraph.prebuilt.tool_node.ToolNode].
    :::

    :::js
    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { ToolNode } from "@langchain/langgraph/prebuilt";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const multiply = tool(
      (input) => {
        if (input.a === 42) {
          throw new Error("The ultimate error");
        }
        return input.a * input.b;
      },
      {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
          a: z.number(),
          b: z.number(),
        }),
      }
    );

    // highlight-next-line
    const toolNode = new ToolNode(
      [multiply],
      {
        // highlight-next-line
        handleToolErrors: "Can't use 42 as a first operand, you must switch operands!"  // (1)!
      }
    );

    const agentCustomErrorHandling = createReactAgent({
      llm: new ChatAnthropic({ model: "claude-3-7-sonnet-latest" }),
      tools: toolNode
    });

    await agentCustomErrorHandling.invoke({
      messages: [{ role: "user", content: "what's 42 x 7?" }]
    });
    ```

    1. This provides a custom message to send to the LLM in case of an exception. See all available strategies in the [API reference][toolnode].
    :::

:::python
See [API reference][langgraph.prebuilt.tool_node.ToolNode] for more information on different tool error handling options.
:::

:::js
See [API reference][toolnode] for more information on different tool error handling options.
:::

## Working with memory

LangGraph allows access to short-term and long-term memory from tools. See [Memory](../how-tos/memory/add-memory.md) guide for more information on:

- how to [read](../how-tos/memory/add-memory.md#read-short-term) from and [write](../how-tos/memory/add-memory.md#write-short-term) to **short-term** memory
- how to [read](../how-tos/memory/add-memory.md#read-long-term) from and [write](../how-tos/memory/add-memory.md#write-long-term) to **long-term** memory

## Prebuilt tools

:::python
You can use prebuilt tools from model providers by passing a dictionary with tool specs to the `tools` parameter of `create_react_agent`. For example, to use the `web_search_preview` tool from OpenAI:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[{"type": "web_search_preview"}]
)
response = agent.invoke(
    {"messages": ["What was a positive news story from today?"]}
)
```

Additionally, LangChain supports a wide range of prebuilt tool integrations for interacting with APIs, databases, file systems, web data, and more. These tools extend the functionality of agents and enable rapid development.

You can browse the full list of available integrations in the [LangChain integrations directory](https://python.langchain.com/docs/integrations/tools/).
:::

:::js
You can use prebuilt tools from model providers by passing a dictionary with tool specs to the `tools` parameter of `createReactAgent`. For example, to use the `web_search_preview` tool from OpenAI:

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const agent = createReactAgent({
  llm: new ChatAnthropic({ model: "claude-3-7-sonnet-latest" }),
  tools: [{ type: "web_search_preview" }],
});

const response = await agent.invoke({
  messages: [
    { role: "user", content: "What was a positive news story from today?" },
  ],
});
```

Additionally, LangChain supports a wide range of prebuilt tool integrations for interacting with APIs, databases, file systems, web data, and more. These tools extend the functionality of agents and enable rapid development.

You can browse the full list of available integrations in the [LangChain integrations directory](https://js.langchain.com/docs/integrations/tools/).
:::

Some commonly used tool categories include:

- **Search**: Bing, SerpAPI, Tavily
- **Code interpreters**: Python REPL, Node.js REPL
- **Databases**: SQL, MongoDB, Redis
- **Web data**: Web scraping and browsing
- **APIs**: OpenWeatherMap, NewsAPI, and others

These integrations can be configured and added to your agents using the same `tools` parameter shown in the examples above.
