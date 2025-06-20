---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Context

Agents often require more than a list of messages to function effectively. They need **context**.

Context includes *any* data outside the message list that can shape agent behavior or tool execution. This can be:

- Information passed at runtime, like a `user_id` or API credentials.
- Internal state updated during a multi-step reasoning process.
- Persistent memory or facts from previous interactions.

LangGraph provides **three** primary ways to supply context:

| Type                                                                         | Description                                   | Mutable? | Lifetime                |
|------------------------------------------------------------------------------|-----------------------------------------------|----------|-------------------------|
| [**Config**](#config-static-context)                                         | data passed at the start of a run             | ❌        | per run                 |
| [**State**](#state-mutable-context)                                          | dynamic data that can change during execution | ✅        | per run or conversation |
| [**Long-term Memory (Store)**](#long-term-memory-cross-conversation-context) | data that can be shared between conversations | ✅        | across conversations    |

You can use context to:

- Adjust the system prompt the model sees
- Feed tools with necessary inputs
- Track facts during an ongoing conversation

## Providing Runtime Context

Use this when you need to inject data into an agent at runtime.

### Config (static context)

Config is for immutable data like user metadata or API keys. Use
when you have values that don't change mid-run.

Specify configuration using a key called **"configurable"** which is reserved
for this purpose:


:::python
```python
agent.invoke(
    {"messages": [{"role": "user", "content": "hi!"}]},
    # highlight-next-line
    config={"configurable": {"user_id": "user_123"}}
)
```
:::

:::js
```ts
await agent.invoke(
  { messages: "hi!" },
  // highlight-next-line
  { configurable: { userId: "user_123" } }
)
```
:::

### State (mutable context)

State acts as short-term memory during a run. It holds dynamic data that can evolve during execution, such as values derived from tools or LLM outputs.

:::python
```python
class CustomState(AgentState):
    # highlight-next-line
    user_name: str

agent = create_react_agent(
    # Other agent parameters...
    # highlight-next-line
    state_schema=CustomState,
)

agent.invoke({
    "messages": "hi!",
    "user_name": "Jane"
})
```
:::

:::js
```ts
const CustomState = Annotation.Root({
    ...MessagesAnnotation.spec,
    userName: Annotation<string>,
});

const agent = createReactAgent({
    // Other agent parameters...
    // highlight-next-line
    stateSchema: CustomState,
})

await agent.invoke(
    // highlight-next-line
    { messages: "hi!", userName: "Jane" }
)
```
:::



!!! tip "Turning on memory"

    Please see the [memory guide](./memory.md) for more details on how to enable memory. This is a powerful feature that allows you to persist the agent's state across multiple invocations.
    Otherwise, the state is scoped only to a single agent run.



### Long-Term Memory (cross-conversation context)

For context that spans *across* conversations or sessions, LangGraph allows access to **long-term memory** via a `store`. This can be used to read or update persistent facts (e.g., user profiles, preferences, prior interactions). For more, see the [Memory guide](./memory.md).

## Customizing Prompts with Context { #prompts }

Prompts define how the agent behaves. To incorporate runtime context, you can dynamically generate prompts based on the agent's state or config.

Common use cases:

- Personalization
- Role or goal customization
- Conditional behavior (e.g., user is admin)

:::python

=== "Using config"

    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState

    def prompt(
        state: AgentState,
        # highlight-next-line
        config: RunnableConfig,
    ) -> list[AnyMessage]:
        # highlight-next-line
        user_name = config["configurable"].get("user_name")
        system_msg = f"You are a helpful assistant. User's name is {user_name}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        # highlight-next-line
        prompt=prompt
    )

    agent.invoke(
        ...,
        # highlight-next-line
        config={"configurable": {"user_name": "John Smith"}}
    )
    ```

=== "Using state"

    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState

    class CustomState(AgentState):
        # highlight-next-line
        user_name: str

    def prompt(
        # highlight-next-line
        state: CustomState
    ) -> list[AnyMessage]:
        # highlight-next-line
        user_name = state["user_name"]
        system_msg = f"You are a helpful assistant. User's name is {user_name}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[...],
        # highlight-next-line
        state_schema=CustomState,
        # highlight-next-line
        prompt=prompt
    )

    agent.invoke({
        "messages": "hi!",
        # highlight-next-line
        "user_name": "John Smith"
    })
    ```

:::

:::js
=== "Using config"

    ```ts
    import { BaseMessageLike } from "@langchain/core/messages";
    import { RunnableConfig } from "@langchain/core/runnables";
    import { initChatModel } from "langchain/chat_models/universal";
    import { MessagesAnnotation } from "@langchain/langgraph";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const prompt = (
      state: typeof MessagesAnnotation.State,
      // highlight-next-line
      config: RunnableConfig
    ): BaseMessageLike[] => {
      // highlight-next-line
      const userName = config.configurable?.userName;
      const systemMsg = `You are a helpful assistant. Address the user as ${userName}.`;
      return [{ role: "system", content: systemMsg }, ...state.messages];
    };

    const llm = await initChatModel("anthropic:claude-3-7-sonnet-latest");
    const agent = createReactAgent({
      llm,
      tools: [getWeather],
      // highlight-next-line
      prompt
    });

    await agent.invoke(
      { messages: "hi!" },
      // highlight-next-line
      { configurable: { userName: "John Smith" } }
    );
    ```

=== "Using state"

    ```ts
    import { BaseMessageLike } from "@langchain/core/messages";
    import { RunnableConfig } from "@langchain/core/runnables";
    import { initChatModel } from "langchain/chat_models/universal";
    import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const CustomState = Annotation.Root({
      ...MessagesAnnotation.spec,
      // highlight-next-line
      userName: Annotation<string>,
    });

    const prompt = (
      // highlight-next-line
      state: typeof CustomState.State,
    ): BaseMessageLike[] => {
      // highlight-next-line
      const userName = state.userName;
      const systemMsg = `You are a helpful assistant. Address the user as ${userName}.`;
      return [{ role: "system", content: systemMsg }, ...state.messages];
    };

    const llm = await initChatModel("anthropic:claude-3-7-sonnet-latest");
    const agent = createReactAgent({
      llm,
      tools: [getWeather],
      // highlight-next-line
      prompt,
      // highlight-next-line
      stateSchema: CustomState,
    });

    await agent.invoke(
      // highlight-next-line
      { messages: "hi!", userName: "John Smith" },
    );
    ```
:::


## Accessing Context in Tools { #tools }

:::python
Tools can access context through special parameter **annotations**.

* Use `RunnableConfig` for config access
* Use `Annotated[StateSchema, InjectedState]` for agent state


!!! tip 

    These annotations prevent LLMs from attempting to fill in the values. These parameters will be **hidden** from the LLM.

=== "Using config"

    ```python
    def get_user_info(
        # highlight-next-line
        config: RunnableConfig,
    ) -> str:
        """Look up user info."""
        # highlight-next-line
        user_id = config["configurable"].get("user_id")
        return "User is John Smith" if user_id == "user_123" else "Unknown user"

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_user_info],
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "look up user information"}]},
        # highlight-next-line
        config={"configurable": {"user_id": "user_123"}}
    )
    ```

=== "Using State"

    ```python
    from typing import Annotated
    from langgraph.prebuilt import InjectedState

    class CustomState(AgentState):
        # highlight-next-line
        user_id: str

    def get_user_info(
        # highlight-next-line
        state: Annotated[CustomState, InjectedState]
    ) -> str:
        """Look up user info."""
        # highlight-next-line
        user_id = state["user_id"]
        return "User is John Smith" if user_id == "user_123" else "Unknown user"

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_user_info],
        # highlight-next-line
        state_schema=CustomState,
    )

    agent.invoke({
        "messages": "look up user information",
        # highlight-next-line
        "user_id": "user_123"
    })
    ```
:::

:::js
Tools can access context through:

* Use `RunnableConfig` for config access
* Use `getCurrentTaskInput()` for agent state

=== "Using config"

    ```ts
    import { RunnableConfig } from "@langchain/core/runnables";
    import { initChatModel } from "langchain/chat_models/universal";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const getUserInfo = tool(
      async (input: Record<string, any>, config: RunnableConfig) => {
        // highlight-next-line
        const userId = config.configurable?.userId;
        return userId === "user_123" ? "User is John Smith" : "Unknown user";
      },
      {
        name: "get_user_info",
        description: "Look up user info.",
        schema: z.object({}),
      }
    );

    const llm = await initChatModel("anthropic:claude-3-7-sonnet-latest");
    const agent = createReactAgent({
      llm,
      tools: [getUserInfo],
    });

    await agent.invoke(
      { messages: "look up user information" },
      // highlight-next-line
      { configurable: { userId: "user_123" } }
    );
    ```

=== "Using state"

    ```ts
    import { initChatModel } from "langchain/chat_models/universal";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { Annotation, MessagesAnnotation, getCurrentTaskInput } from "@langchain/langgraph";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const CustomState = Annotation.Root({
      ...MessagesAnnotation.spec,
      // highlight-next-line
      userId: Annotation<string>(),
    });

    const getUserInfo = tool(
      async (
        input: Record<string, any>,
      ) => {
        // highlight-next-line
        const state = getCurrentTaskInput() as typeof CustomState.State;
        // highlight-next-line
        const userId = state.userId;
        return userId === "user_123" ? "User is John Smith" : "Unknown user";
      },
      {
        name: "get_user_info",
        description: "Look up user info.",
        schema: z.object({})
      }
    );

    const llm = await initChatModel("anthropic:claude-3-7-sonnet-latest");
    const agent = createReactAgent({
      llm,
      tools: [getUserInfo],
      // highlight-next-line
      stateSchema: CustomState,
    });

    await agent.invoke(
      // highlight-next-line
      { messages: "look up user information", userId: "user_123" }
    );
    ```
:::

### Update Context from Tools

:::python
Tools can update agent's context (state and long-term memory) during execution. This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts. See [Memory](./memory.md#read-short-term) guide for more information.
:::

:::js
Tools can modify the agent's state during execution. This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.

```ts
import { Annotation, MessagesAnnotation, LangGraphRunnableConfig, Command } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ToolMessage } from "@langchain/core/messages";
import { initChatModel } from "langchain/chat_models/universal";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const CustomState = Annotation.Root({
  ...MessagesAnnotation.spec,
  // highlight-next-line
  userName: Annotation<string>(), // Will be updated by the tool
});

const getUserInfo = tool(
  async (
    _input: Record<string, never>,
    config: LangGraphRunnableConfig
  ): Promise<Command> => {
    const userId = config.configurable?.userId;
    if (!userId) {
      throw new Error("Please provide a user id in config.configurable");
    }

    const toolCallId = config.toolCall?.id;

    const name = userId === "user_123" ? "John Smith" : "Unknown user";
    // Return command to update state
    return new Command({
      update: {
        // highlight-next-line
        userName: name,
        // Update the message history
        // highlight-next-line
        messages: [
          new ToolMessage({
            content: "Successfully looked up user information",
            tool_call_id: toolCallId,
          }),
        ],
      },
    });
  },
  {
    name: "get_user_info",
    description: "Look up user information.",
    schema: z.object({}),
  }
);

const llm = await initChatModel("anthropic:claude-3-7-sonnet-latest");
const agent = createReactAgent({
  llm,
  tools: [getUserInfo],
  // highlight-next-line
  stateSchema: CustomState,
});

await agent.invoke(
  { messages: "look up user information" },
  // highlight-next-line
  { configurable: { userId: "user_123" } }
);
```
:::