# How to Build Interruptible LangGraph Agent With Copilotkit

!!! info "Prerequisites"

    - [LangGraph Server](../../concepts/langgraph_server.md)

[CopilotKit](https://copilotkit.ai/) makes it easy to build human-in-the-loop workflows with LangGraph. It provides simple tools to handle interruptions, get user input, and smoothly include human decisions in your agent's flow.

## Installation

```bash
npm install @copilotkit/react-core @copilotkit/react-ui @copilotkit/runtime
```

## Basic Setup

Integrating LangGraph with CopilotKit involves two main steps: setting up the backend runtime and configuring your frontend components.

### Copilot Runtime

```ts
import { createServer } from "node:http";
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNodeHttpEndpoint,
  langGraphPlatformEndpoint,
} from "@copilotkit/runtime";

const serviceAdapter = new ExperimentalEmptyAdapter();

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
    endpoint: "/copilotkit",
    runtime,
    serviceAdapter,
  });

  return handler(req, res);
});

server.listen(4000, () => {
  console.log("Listening at http://localhost:4000/copilotkit");
});
```

### Frontend

```tsx
// layout.tsx
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotTextarea } from "@copilotkit/react-ui";

function MyApp({ Component, pageProps }) {
  return <CopilotKit runtimeUrl="/api/copilotkit">{children}</CopilotKit>;
}

import { CopilotChat } from "@copilotkit/react-ui";

export function YourComponent() {
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

## Core Concepts

### Interrupts

Interrupts allow you to pause agent execution to get user input. With CopilotKit, you can:

- Create interactive flows that require user decisions
- Present custom UI elements during interrupts
- Make your agent aware of interrupt interactions
- Handle multiple types of interrupts

#### Basic Interrupt Usage

To use interrupts in your agent:

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

By default, agents don't see interrupt interactions in their context. To include them:

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

Handle multiple interrupt types with the `enabled` property:

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

#### Advanced Preprocessing

Use the `handler` property to preprocess interrupts before rendering UI:

```tsx
useLangGraphInterrupt({
  handler: async ({ event, resolve }) => {
    // Process interrupt data
    if (canAutoResolve(event.value)) {
      resolve(computedResponse);
      return;
    }
    return processedData;
  },
  render: ({ result, event, resolve }) => (
    <CustomComponent data={result} onSubmit={(value) => resolve(value)} />
  ),
});
```

## Further Reading

- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [Interrupt](https://docs.copilotkit.ai/coagents/human-in-the-loop/interrupt-flow)
