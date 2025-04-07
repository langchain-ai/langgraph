---
title: Using CopilotKit with LangGraph
description: Turn your LangGraph agent into an agent-native application in minutes
---

# How to Use CopilotKit with LangGraph

CopilotKit is an open-source framework for building AI copilots in web applications. This guide will show you how to connect your LangGraph agent to a React application using CopilotKit.

## Prerequisites

- An existing LangGraph agent (running locally with LangGraph Studio)
- LangSmith API key (optional)

## Installation

### Frontend

```bash
npm install @copilotkit/react-ui @copilotkit/react-core
```

### Backend

```bash
npm install @copilotkit/runtime
```

## Setup

### 1. Start your LangGraph agent

Run your LangGraph agent locally using LangGraph Studio:

```bash
langgraph dev
```

This will start your agent on the default port (8000).

### 2. Create a Copilot Runtime endpoint

Create a simple Node.js server file to handle requests:

```javascript
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
        deploymentUrl: "http://localhost:8000", // make sure to replace with your real deployment url,
        langsmithApiKey: process.env.LANGSMITH_API_KEY, // only used in LangGraph Platform deployments
        agents: [
          // List any agents available under "graphs" list in your langgraph.json file; give each a description explaining when it should be called.
          {
            name: "sample_agent",
            description: "A helpful LLM agent.",
            assistantId: "your-assistant-ID", // optional, but recommended!
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

Start your Copilot Runtime server:

```bash
node copilot-server.js
```

### 3. Set up the CopilotKit Provider in your React app

Wrap your application with the CopilotKit Provider:

```jsx
import { CopilotKit } from "@copilotkit/react-core";

function App() {
  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="sample_agent" // the name of the agent you want to use
    >
      <YourApp />
    </CopilotKit>
  );
}
```

### 4. Add a Copilot UI component

CopilotKit provides ready-to-use UI components. Add a chat interface to your app:

```jsx
import { CopilotChat } from "@copilotkit/react-ui";

function MyComponent() {
  return (
    <div>
      {/* Your app content */}

      {/* Floating chat interface */}
      <CopilotChat
        instructions={
          "You are assisting the user as best as you can. Answer in the best way possible given the data you have."
        }
        labels={{
          title: "Your Assistant",
          initial: "Hi! 👋 How can I assist you today?",
        }}
      />
    </div>
  );
}
```

## Testing your integration

Start your React application and talk to your agent! Try asking:

```
What can you help me with?
```

## Next steps

You've added an agent to your existing app using CopilotKit. Here are some advanced features you can explore:

- **Human-in-the-loop**: Enable collaboration between users and agents
- **Shared state**: Synchronize agent state with UI state
- **Generative UI**: Render agent outputs directly in your interface
- **Frontend actions**: Allow agents to call frontend functions

For more detailed guides and advanced options, refer to the [official CopilotKit documentation](https://docs.copilotkit.ai/coagents/quickstart/langgraph).
