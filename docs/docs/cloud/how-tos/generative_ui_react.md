# How to implement Generative User Interfaces with LangGraph

!!! info "Prerequisites"

    - [LangGraph Platform](../../concepts/langgraph_platform.md)
    - [LangGraph Server](../../concepts/langgraph_server.md)
    - [`useStream()` React Hook](./use_stream_react.md)

Generative user interfaces (Generative UI) is a powerful pattern that allows AI agents to go beyond text and generate rich user interfaces. This enables creating more interactive and context-aware applications where the UI adapts based on the conversation flow and AI responses.

LangGraph Platform supports colocating your UI React components with your graph code. This allows you to focus on building specific UI components for your graph while easily plugging into existing chat interfaces such as [Agent Chat](https://agentchat.vercel.app) and loading the code only when actually needed.

## Quickstart

### 1. Define and configure UI components

First, create your first UI component. For each component you need to provide an unique identifier that will be used to reference the component in your graph code.

```tsx
// src/agent/ui.tsx
const WeatherComponent = (props: { city: string }) => {
  return <div>Weather for {props.city}</div>;
};

export default {
  weather: WeatherComponent,
};
```

Next, define your UI components in your `langgraph.json` configuration:

```json
{
  "node_version": "20",
  "graphs": {
    "agent": "./src/agent/index.ts:graph"
  },
  "ui": {
    "agent": "./src/agent/ui.tsx"
  }
}
```

The `ui` section points to the UI components that will be used by the respective graph. LangGraph Platform supports React 18.x and above, as well as Tailwind 4.x. Dependencies will be bundled and served from the LangGraph Platform.

### 2. Send the UI components in your graph

Use the `typedUi` utility to emit UI elements from your agent nodes:

```typescript
// src/agent/index.ts
import type ComponentMap from "./ui";
import {
  typedUi,
  uiMessageReducer,
} from "@langchain/langgraph-sdk/react-ui/server";

import {
  Annotation,
  MessagesAnnotation,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";

import { v4 as uuidv4 } from "uuid";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  ui: Annotation({ default: () => [], reducer: uiMessageReducer }),
});

export async function getWeather(
  state: typeof AgentState,
  config: LangGraphRunnableConfig
) {
  // Provide the type of the component map to ensure
  // type safety of `ui.push()` calls.
  const ui = typedUi<typeof ComponentMap>(config);

  const weather = await new ChatOpenAI({ model: "gpt-4o-mini" })
    .withStructuredOutput(z.object({ city: z.string() }))
    .invoke(state.messages);

  const response = {
    id: uuidv4(),
    type: "ai",
    content: `Here's the weather for ${weather.city}`,
  };

  // Emit UI elements with associated AI message
  ui.push({ name: "weather", props: weather }, { message: ai });

  return {
    messages: [response],
    ui: ui.items,
  };
}
```

### 3. Handle UI elements in your React application

On the client side, you can use `useStream()` and `LoadExternalComponent` to display the UI elements.

```tsx
// src/app/page.tsx
"use client";

import { useStream } from "@langchain/langgraph-sdk/react";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui/client";

export default function Page() {
  const { thread, values } = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
  });

  return (
    <div>
      {thread.messages.map((message) => (
        <div key={message.id}>
          {message.content}
          {values.ui
            ?.filter((ui) => ui.metadata?.message_id === message.id)
            .map((ui) => (
              <LoadExternalComponent key={ui.id} stream={thread} message={ui} />
            ))}
        </div>
      ))}
    </div>
  );
}
```

## Configuration options

### Show loading UI when components are loading

You can provide a fallback UI to be rendered when the components are loading.

```tsx
<LoadExternalComponent
  stream={thread}
  message={ui}
  fallback={<div>Loading...</div>}
/>
```

### Provide custom components on the client side

If you already have the components loaded in your client application, you can provide a map of such components to be rendered directly without fetching the UI code from LangGraph Platform.

```tsx
const clientComponents = {
  weather: WeatherComponent,
};

<LoadExternalComponent
  stream={thread}
  message={ui}
  components={clientComponents}
/>;
```

### Access the thread state from the UI component

You can access the thread state from the UI component by using the `useStreamContext` hook.

```tsx
const WeatherComponent = (props: { city: string }) => {
  const { thread, submit } = useStreamContext();
  return (
    <>
      <div>Weather for {props.city}</div>

      <button
        onClick={() => {
          const newMessage = {
            type: "human",
            content: `What's the weather in ${props.city}?`,
          };

          submit({ messages: [newMessage] });
        }}
      >
        Retry
      </button>
    </>
  );
};
```

### Pass additional context to the client components

You can pass additional context to the client components by providing a `meta` prop to the `LoadExternalComponent` component.

```tsx
<LoadExternalComponent stream={thread} message={ui} meta={{ userId: "123" }} />
```

Then, you can access the `meta` prop in the UI component by using the `useStreamContext` hook.

```tsx
const WeatherComponent = (props: { city: string }) => {
  const { meta } = useStreamContext();
  return (
    <div>
      Weather for {props.city} (user: {meta?.userId})
    </div>
  );
};
```

### Streaming UI updates before the node execution is finished

You can stream UI updates before the node execution is finished by using the `onCustomEvent` callback of the `useStream()` hook.

```tsx
import { uiMessageReducer } from "@langchain/langgraph-sdk/react-ui";

const { thread, submit } = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  onCustomEvent: (event, options) => {
    options.mutate((prev) => {
      const ui = uiMessageReducer(prev.ui ?? [], event);
      return { ...prev, ui };
    });
  },
});
```

## Learn more

- [JS/TS SDK Reference](../reference/sdk/js_ts_sdk_ref.md)
