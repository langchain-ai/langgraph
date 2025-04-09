# How to use LangGraph with CopilotKit in React

!!! info "Prerequisites"

    - [LangGraph Server](../../concepts/langgraph_server.md)

CopilotKit provides React components to quickly integrate customizable AI copilots into your application. Combined with LangGraph, you can build sophisticated AI apps featuring bidirectional state synchronization and interactive UIs.

## Installation

```bash
# Frontend packages
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

### Shared State

The `useCoAgent` hook enables shared state management between your React application and LangGraph agent, allowing bidirectional synchronization. You can:

- Read the agent's current state in your UI components
- Update the agent's state from your UI
- Trigger agent reruns with updated state

#### Defining Agent State

```ts
import { Annotation } from "@langchain/langgraph";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";

export const AgentStateAnnotation = Annotation.Root({
  language: Annotation<"english" | "spanish">,
  ...CopilotKitStateAnnotation.spec,
});
export type AgentState = typeof AgentStateAnnotation.State;

async function chat_node(state: AgentState, config: RunnableConfig) {
  // Use language from state or default to spanish
  const language = state.language ?? "spanish";

  // ... node implementation

  return {
    // Return language to make it available for next nodes & frontend
    language,
  };
}
```

#### Reading and Writing Agent State

The [`useCoAgent`](https://docs.copilotkit.ai/reference/hooks/useCoAgent) hook provides both state access and update capabilities:

```ts
import { useCoAgent } from "@copilotkit/react-core";

// Define state type matching your agent's state
type AgentState = {
  language: "english" | "spanish";
};

function YourComponent() {
  const { state, setState } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: { language: "spanish" },
  });

  const toggleLanguage = () => {
    setState({
      language: state.language === "english" ? "spanish" : "english",
    });
  };

  return (
    <div>
      <p>Current language: {state.language}</p>
      <button onClick={toggleLanguage}>Toggle Language</button>
    </div>
  );
}
```

The state in useCoAgent updates automatically when the agent's state changes.

#### Rendering State in Chat UI

You can also render the agent's state directly in the chat interface using [`useCoAgentStateRender`](https://docs.copilotkit.ai/reference/hooks/useCoAgentStateRender):

```ts
import { useCoAgentStateRender } from "@copilotkit/react-core";

function YourComponent() {
  // Render agent state in the chat UI
  useCoAgentStateRender({
    name: "sample_agent",
    render: ({ state }) => {
      if (!state.language) return null;
      return <div>Language: {state.language}</div>;
    },
  });

  // Rest of component
}
```

#### Running the Agent Flow with `run`

Use the `run` function to trigger the agent flow again whenever the state changes:

```tsx
import { useCoAgent } from "@copilotkit/react-core";
import { TextMessage, MessageRole } from "@copilotkit/runtime-client-gql";

function YourComponent() {
  const { state, setState, run } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: { language: "spanish" },
  });

  const toggleLanguage = () => {
    const newLanguage = state.language === "english" ? "spanish" : "english";
    setState({ language: newLanguage });

    // Rerun with a hint about what changed
    run(({ currentState }) => {
      return new TextMessage({
        role: MessageRole.User,
        content: `The language has been updated to ${currentState.language}`,
      });
    });
  };

  // Component JSX
}
```

By default, state updates only occur during node transitions. For continuous state updates, see the CopilotKit documentation on streaming intermediate state.

### Frontend Actions

Frontend actions enable your LangGraph agent to interact with and update your application's UI. They serve as bridges connecting your agent's logic to the interactive elements of your frontend.

With frontend actions, your agent can:

- Update UI elements dynamically
- Trigger frontend animations or transitions
- Show alerts or notifications
- Modify application state
- Handle user interactions programmatically

#### Creating a Frontend Action

```tsx
import { useCopilotAction } from "@copilotkit/react-core";

export function Page() {
  useCopilotAction({
    name: "sayHello",
    description: "Say hello to the user",
    available: "remote", // makes the action only available to the agent
    parameters: [
      {
        name: "name",
        type: "string",
        description: "The name of the user to say hello to",
        required: true,
      },
    ],
    handler: async ({ name }) => {
      alert(`Hello, ${name}!`);
    },
  });

  // Rest of component
}
```

This hook creates an action that the agent can trigger to interact with your UI.

#### When should I use this?

Frontend actions are essential when you want to create truly interactive AI applications where your agent needs to:

- Dynamically update UI elements
- Trigger frontend animations or transitions
- Show alerts or notifications
- Modify application state
- Handle user interactions programmatically

#### Implementation

First, you'll need to create a frontend action using the useCopilotAction hook. Here's a simple one to get you started that says hello to the user.

##### Install the CopilotKit SDK

Any LangGraph agent can be used with CopilotKit. However, creating deep agentic experiences with CopilotKit requires our LangGraph SDK.

```
npm install @copilotkit/sdk-js
```

##### Inheriting from CopilotKitState

To access the frontend actions provided by CopilotKit, you can inherit from CopilotKitState in your agent's state definition:

```ts
import { Annotation } from "@langchain/langgraph";
import { CopilotKitStateAnnotation } from "@copilotkit/sdk-js/langgraph";

export const YourAgentStateAnnotation = Annotation.Root({
  yourAdditionalProperty: Annotation<string>,
  ...CopilotKitStateAnnotation.spec,
});
export type YourAgentState = typeof YourAgentStateAnnotation.State;
```

##### Accessing Frontend Actions

Once your agent's state includes the copilotkit property, you can access the frontend actions and utilize them within your agent's logic.

Here's how you can call a frontend action from your agent:

```ts
async function agentNode(
  state: YourAgentState,
  config: RunnableConfig
): Promise<YourAgentState> {
  // Access the actions from the copilotkit property

  const actions = state.copilotkit?.actions;
  const model = ChatOpenAI({ model: "gpt-4o" }).bindTools(actions);

  // ...
}
```

CopilotKit automatically provides these actions as LangChain-compatible tools, letting your agent easily call them as needed.

## Further Reading

- [CopilotKit Documentation](https://docs.copilotkit.ai)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [React Hooks with CopilotKit](https://docs.copilotkit.ai/reference/hooks/useCoAgent)
