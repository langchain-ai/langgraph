---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Multi-agent

A single agent might struggle if it needs to specialize in multiple domains or manage many tools. To tackle this, you can break your agent into smaller, independent agents and compose them into a [multi-agent system](../concepts/multi_agent.md).

In multi-agent systems, agents need to communicate between each other. They do so via [handoffs](#handoffs) — a primitive that describes which agent to hand control to and the payload to send to that agent.

Two of the most popular multi-agent architectures are:

- [supervisor](#supervisor) — individual agents are coordinated by a central supervisor agent. The supervisor controls all communication flow and task delegation, making decisions about which agent to invoke based on the current context and task requirements.
- [swarm](#swarm) — agents dynamically hand off control to one another based on their specializations. The system remembers which agent was last active, ensuring that on subsequent interactions, the conversation resumes with that agent.

## Supervisor

![Supervisor](./assets/supervisor.png)

:::python
Use [`langgraph-supervisor`](https://github.com/langchain-ai/langgraph-supervisor-py) library to create a supervisor multi-agent system:

```bash
pip install langgraph-supervisor
```

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_supervisor import create_supervisor

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

flight_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_flight],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# highlight-next-line
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

:::

:::js
Use [`@langchain/langgraph-supervisor`](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-supervisor) library to create a supervisor multi-agent system:

```bash
npm install @langchain/langgraph-supervisor
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
// highlight-next-line
import { createSupervisor } from "langgraph-supervisor";

function bookHotel(hotelName: string) {
  /**Book a hotel*/
  return `Successfully booked a stay at ${hotelName}.`;
}

function bookFlight(fromAirport: string, toAirport: string) {
  /**Book a flight*/
  return `Successfully booked a flight from ${fromAirport} to ${toAirport}.`;
}

const flightAssistant = createReactAgent({
  llm: "openai:gpt-4o",
  tools: [bookFlight],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: "openai:gpt-4o",
  tools: [bookHotel],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// highlight-next-line
const supervisor = createSupervisor({
  agents: [flightAssistant, hotelAssistant],
  llm: new ChatOpenAI({ model: "gpt-4o" }),
  systemPrompt:
    "You manage a hotel booking assistant and a " +
    "flight booking assistant. Assign work to them.",
});

for await (const chunk of supervisor.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

:::

## Swarm

![Swarm](./assets/swarm.png)

:::python
Use [`langgraph-swarm`](https://github.com/langchain-ai/langgraph-swarm-py) library to create a swarm multi-agent system:

```bash
pip install langgraph-swarm
```

```python
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_swarm import create_swarm, create_handoff_tool

transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# highlight-next-line
swarm = create_swarm(
    agents=[flight_assistant, hotel_assistant],
    default_active_agent="flight_assistant"
).compile()

for chunk in swarm.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

:::

:::js
Use [`@langchain/langgraph-swarm`](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-swarm) library to create a swarm multi-agent system:

```bash
npm install @langchain/langgraph-swarm
```

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
// highlight-next-line
import { createSwarm, createHandoffTool } from "@langchain/langgraph-swarm";

const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
  description: "Transfer user to the hotel-booking assistant.",
});

const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
  description: "Transfer user to the flight-booking assistant.",
});

const flightAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  // highlight-next-line
  tools: [bookFlight, transferToHotelAssistant],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  // highlight-next-line
  tools: [bookHotel, transferToFlightAssistant],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// highlight-next-line
const swarm = createSwarm({
  agents: [flightAssistant, hotelAssistant],
  defaultActiveAgent: "flight_assistant",
});

for await (const chunk of swarm.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

:::

## Handoffs

A common pattern in multi-agent interactions is **handoffs**, where one agent _hands off_ control to another. Handoffs allow you to specify:

- **destination**: target agent to navigate to
- **payload**: information to pass to that agent

:::python
This is used both by `langgraph-supervisor` (supervisor hands off to individual agents) and `langgraph-swarm` (an individual agent can hand off to other agents).

To implement handoffs with `create_react_agent`, you need to:

1.  Create a special tool that can transfer control to a different agent

    ```python
    def transfer_to_bob():
        """Transfer to bob."""
        return Command(
            # name of the agent (node) to go to
            # highlight-next-line
            goto="bob",
            # data to send to the agent
            # highlight-next-line
            update={"messages": [...]},
            # indicate to LangGraph that we need to navigate to
            # agent node in a parent graph
            # highlight-next-line
            graph=Command.PARENT,
        )
    ```

2.  Create individual agents that have access to handoff tools:

    ```python
    flight_assistant = create_react_agent(
        ..., tools=[book_flight, transfer_to_hotel_assistant]
    )
    hotel_assistant = create_react_agent(
        ..., tools=[book_hotel, transfer_to_flight_assistant]
    )
    ```

3.  Define a parent graph that contains individual agents as nodes:

    ```python
    from langgraph.graph import StateGraph, MessagesState
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node(flight_assistant)
        .add_node(hotel_assistant)
        ...
    )
    ```

:::

:::js
This is used both by `@langchain/langgraph-supervisor` (supervisor hands off to individual agents) and `@langchain/langgraph-swarm` (an individual agent can hand off to other agents).

To implement handoffs with `createReactAgent`, you need to:

1.  Create a special tool that can transfer control to a different agent

    ```typescript
    function transferToBob() {
      /**Transfer to bob.*/
      return new Command({
        // name of the agent (node) to go to
        // highlight-next-line
        goto: "bob",
        // data to send to the agent
        // highlight-next-line
        update: { messages: [...] },
        // indicate to LangGraph that we need to navigate to
        // agent node in a parent graph
        // highlight-next-line
        graph: Command.PARENT,
      });
    }
    ```

2.  Create individual agents that have access to handoff tools:

    ```typescript
    const flightAssistant = createReactAgent({
      ..., tools: [bookFlight, transferToHotelAssistant]
    });
    const hotelAssistant = createReactAgent({
      ..., tools: [bookHotel, transferToFlightAssistant]
    });
    ```

3.  Define a parent graph that contains individual agents as nodes:

    ```typescript
    import { StateGraph, MessagesZodState } from "@langchain/langgraph";
    const multiAgentGraph = new StateGraph(MessagesZodState)
      .addNode("flight_assistant", flightAssistant)
      .addNode("hotel_assistant", hotelAssistant)
      // ...
    ```

    :::

Putting this together, here is how you can implement a simple multi-agent system with two agents — a flight booking assistant and a hotel booking assistant:

:::python

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        # highlight-next-line
        state: Annotated[MessagesState, InjectedState], # (1)!
        # highlight-next-line
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(  # (2)!
            # highlight-next-line
            goto=agent_name,  # (3)!
            # highlight-next-line
            update={"messages": state["messages"] + [tool_message]},  # (4)!
            # highlight-next-line
            graph=Command.PARENT,  # (5)!
        )
    return handoff_tool

# Handoffs
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

# Simple agent tools
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

# Define agents
flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# Define multi-agent graph
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)

# Run the multi-agent graph
for chunk in multi_agent_graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

1. Access agent's state
2. The `Command` primitive allows specifying a state update and a node transition as a single operation, making it useful for implementing handoffs.
3. Name of the agent or node to hand off to.
4. Take the agent's messages and **add** them to the parent's **state** as part of the handoff. The next agent will see the parent state.
5. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.
   :::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  StateGraph,
  START,
  MessagesZodState,
  Command,
} from "@langchain/langgraph";
import { z } from "zod";

function createHandoffTool({
  agentName,
  description,
}: {
  agentName: string;
  description?: string;
}) {
  const name = `transfer_to_${agentName}`;
  const toolDescription = description || `Transfer to ${agentName}`;

  return tool(
    async (_, config) => {
      const toolMessage = {
        role: "tool" as const,
        content: `Successfully transferred to ${agentName}`,
        name: name,
        tool_call_id: config.toolCall?.id!,
      };
      return new Command({
        // (2)!
        // highlight-next-line
        goto: agentName, // (3)!
        // highlight-next-line
        update: { messages: [toolMessage] }, // (4)!
        // highlight-next-line
        graph: Command.PARENT, // (5)!
      });
    },
    {
      name,
      description: toolDescription,
      schema: z.object({}),
    }
  );
}

// Handoffs
const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
  description: "Transfer user to the hotel-booking assistant.",
});

const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
  description: "Transfer user to the flight-booking assistant.",
});

// Simple agent tools
const bookHotel = tool(
  async ({ hotelName }) => {
    /**Book a hotel*/
    return `Successfully booked a stay at ${hotelName}.`;
  },
  {
    name: "book_hotel",
    description: "Book a hotel",
    schema: z.object({
      hotelName: z.string().describe("Name of the hotel to book"),
    }),
  }
);

const bookFlight = tool(
  async ({ fromAirport, toAirport }) => {
    /**Book a flight*/
    return `Successfully booked a flight from ${fromAirport} to ${toAirport}.`;
  },
  {
    name: "book_flight",
    description: "Book a flight",
    schema: z.object({
      fromAirport: z.string().describe("Departure airport code"),
      toAirport: z.string().describe("Arrival airport code"),
    }),
  }
);

// Define agents
const flightAssistant = createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }),
  // highlight-next-line
  tools: [bookFlight, transferToHotelAssistant],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }),
  // highlight-next-line
  tools: [bookHotel, transferToFlightAssistant],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// Define multi-agent graph
const multiAgentGraph = new StateGraph(MessagesZodState)
  .addNode("flight_assistant", flightAssistant)
  .addNode("hotel_assistant", hotelAssistant)
  .addEdge(START, "flight_assistant")
  .compile();

// Run the multi-agent graph
for await (const chunk of multiAgentGraph.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

1. Access agent's state
2. The `Command` primitive allows specifying a state update and a node transition as a single operation, making it useful for implementing handoffs.
3. Name of the agent or node to hand off to.
4. Take the agent's messages and **add** them to the parent's **state** as part of the handoff. The next agent will see the parent state.
5. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.

:::

!!! Note

    This handoff implementation assumes that:

    - each agent receives overall message history (across all agents) in the multi-agent system as its input
    - each agent outputs its internal messages history to the overall message history of the multi-agent system

:::python
Check out LangGraph [supervisor](https://github.com/langchain-ai/langgraph-supervisor-py#customizing-handoff-tools) and [swarm](https://github.com/langchain-ai/langgraph-swarm-py#customizing-handoff-tools) documentation to learn how to customize handoffs.
:::

:::js
Check out LangGraph [supervisor](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-supervisor#customizing-handoff-tools) and [swarm](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-swarm#customizing-handoff-tools) documentation to learn how to customize handoffs.
:::
