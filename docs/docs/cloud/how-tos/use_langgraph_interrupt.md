# How to Build Interruptible LangGraph Agent With Copilotkit

!!! info "Prerequisites"

    - [LangGraph Server](../../concepts/langgraph_server.md)

[CopilotKit](https://copilotkit.ai/) makes it easy to build human-in-the-loop workflows with LangGraph. It provides simple tools to handle interruptions, get user input, and smoothly include human decisions in your agent's flow.

## Installation

```bash
# Frontend packages
npm install @copilotkit/react-core @copilotkit/react-ui
```

## Basic Setup

Integrating LangGraph with CopilotKit involves configuring your frontend components and optionally setting up a backend runtime. The fastest way to get started is to use [Copilotkit Cloud](https://cloud.copilotkit.ai/), which provides a stateless interaction layer that enables seamless communication between LangGraph and your frontend.

### Frontend

```tsx
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotTextarea } from "@copilotkit/react-ui";

function MyApp({ Component, pageProps }) {
  return (
    <CopilotKit publicApiKey="ck_xxxxxxxxxxxxxxxxxxxxxxxxxxxx">
      {children}
    </CopilotKit>
  );
}

import { CopilotChat } from "@copilotkit/react-ui";

export function AIAssistantChatContainer() {
  return (
    <CopilotChat
      instructions={
        "You are assisting the user as best as you can. Answer in the best way possible given the data you have."
      }
      labels={{
        title: "Your Assistant",
        initial: "Hi! 👋 How can I assist you today?",
      }}
    />
  );
}
```

### Copilot Runtime

If you're not using Copilotkit Cloud, you'll need to set up a runtime server to connect your frontend with LangGraph. First, install the runtime package:

```bash
npm install @copilotkit/runtime
```

Then create a server file with the following code:

```ts
import { createServer } from "node:http";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNodeHttpEndpoint,
  langGraphPlatformEndpoint,
} from "@copilotkit/runtime";

const server = createServer((req, res) => {
  const runtime = new CopilotRuntime({
    remoteEndpoints: [
      langGraphPlatformEndpoint({
        // make sure to replace with URL that points to your running LangGraph server
        deploymentUrl: "http://localhost:8000",
        langsmithApiKey: process.env.LANGSMITH_API_KEY, // only used in LangGraph Platform deployments
        agents: [
          {
            name: "sample_agent",
            description: "A helpful LLM agent.",
          },
        ],
      }),
    ],
  });

  const handler = copilotRuntimeNodeHttpEndpoint({
    endpoint: "/api/copilotkit",
    runtime,
  });

  return handler(req, res);
});

server.listen(4000, () => {
  console.log("Listening at http://localhost:4000/api/copilotkit");
});
```

Start this server alongside your frontend application, and ensure your frontend's `CopilotKit` component points to this endpoint:

```tsx
<CopilotKit runtimeUrl="http://localhost:4000/api/copilotkit">
  {children}
</CopilotKit>
```

For more detailed information on self-hosting the Copilot Runtime, see the [CopilotKit Self-Hosting Guide](https://docs.copilotkit.ai/guides/self-hosting).

## Core Concepts

### Interrupts

Interrupts in LangGraph allow you to pause agent execution to get user input. Combined with CopilotKit, you can create truly interactive human-in-the-loop experiences by:

- Creating decision points that require user approval or input
- Presenting custom UI elements during interrupt moments
- Making your agent aware of interrupt interactions
- Supporting different types of interrupts with customized handling

#### Basic Interrupt Usage

To use interrupts in your LangGraph agent:

```ts
import { interrupt } from "@langchain/langgraph";

async function chat_node(state: AgentState, config: RunnableConfig) {
  if (!state.agentName) {
    state.agentName = await interrupt(
      "Before we start, what would you like to call me?"
    );
  }

  // Continue with agent logic using the agentName
  // ...
}
```

#### Handling Interrupts in UI

The [`useLangGraphInterrupt`](https://docs.copilotkit.ai/reference/hooks/useLangGraphInterrupt) hook renders UI components when an interrupt occurs:

```tsx
import { useLangGraphInterrupt } from "@copilotkit/react-core";

function YourComponent() {
  useLangGraphInterrupt({
    render: ({ event, resolve }) => (
      <div>
        <p>{event.value}</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            resolve((e.target as HTMLFormElement).response.value);
          }}
        >
          <input type="text" name="response" />
          <button type="submit">Submit</button>
        </form>
      </div>
    ),
  });

  // Rest of component
}
```

#### Making Agents Aware of Interrupts

By default, agents don't see interrupt interactions in their context. To include them in the message history:

```ts
import { copilotKitInterrupt } from "@copilotkit/sdk-js/langgraph";

async function chat_node(state: AgentState, config: RunnableConfig) {
  const { agentName, newMessages } = await copilotKitInterrupt(
    "Before we start, what would you like to call me?"
  );

  state.messages = [...state.messages, ...newMessages];
  state.agentName = agentName;

  // Continue with agent logic
}
```

#### Conditional Interrupts

To handle multiple interrupt types with different UI components:

```ts
// In your agent
state.approval = await interrupt({
  type: "approval",
  content: "Please approve this action",
});
state.name = await interrupt({ type: "ask", content: "What's your name?" });

// In your UI component
useLangGraphInterrupt({
  enabled: ({ eventValue }) => eventValue.type === "approval",
  render: ({ event, resolve }) => (
    <ApprovalComponent
      onApprove={() => resolve(true)}
      onReject={() => resolve(false)}
    />
  ),
});

useLangGraphInterrupt({
  enabled: ({ eventValue }) => eventValue.type === "ask",
  render: ({ event, resolve }) => (
    <InputComponent
      question={event.value.content}
      onSubmit={(answer) => resolve(answer)}
    />
  ),
});
```

## Further Reading

- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Human-in-the-Loop with Interrupts](https://docs.copilotkit.ai/coagents/human-in-the-loop/interrupt-flow)
- [LangGraph Human-in-the-Loop Guide](../../concepts/human_in_the_loop.md)
