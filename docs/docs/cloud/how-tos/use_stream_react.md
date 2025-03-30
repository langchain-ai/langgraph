# How to integrate LangGraph into your React application

!!! info "Prerequisites" - [LangGraph Platform](../../concepts/langgraph_platform.md) - [LangGraph Server](../../concepts/langgraph_server.md)

The `useStream()` React hook provides a seamless way to integrate LangGraph into your React applications. It handles all the complexities of streaming, state management, and branching logic, letting you focus on building great chat experiences.

Key features:

- Messages streaming: Handle a stream of message chunks to form a complete message
- Automatic state management for messages, interrupts, loading states, and errors
- Conversation branching: Create alternate conversation paths from any point in the chat history
- UI-agnostic design: bring your own components and styling

Let's explore how to use `useStream()` in your React application.

The `useStream()` provides a solid foundation for creating bespoke chat experiences. For pre-built chat components and interfaces, we also recommend checking out [CopilotKit](https://docs.copilotkit.ai/coagents/quickstart/langgraph) and [assistant-ui](https://www.assistant-ui.com/docs/runtimes/langgraph).

## Installation

```bash
npm install @langchain/langgraph-sdk @langchain/core
```

## Example

```tsx
"use client";

import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";

export default function App() {
  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      <div>
        {thread.messages.map((message) => (
          <div key={message.id}>{message.content as string}</div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();

          const form = e.target as HTMLFormElement;
          const message = new FormData(form).get("message") as string;

          form.reset();
          thread.submit({ messages: [{ type: "human", content: message }] });
        }}
      >
        <input type="text" name="message" />

        {thread.isLoading ? (
          <button key="stop" type="button" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button keytype="submit">Send</button>
        )}
      </form>
    </div>
  );
}
```

## Customizing Your UI

The `useStream()` hook takes care of all the complex state management behind the scenes, providing you with simple interfaces to build your UI. Here's what you get out of the box:

- Thread state management
- Loading and error states
- Interrupts
- Message handling and updates
- Branching support

Here are some examples on how to use these features effectively:

### Loading States

The `isLoading` property tells you when a stream is active, enabling you to:

- Show a loading indicator
- Disable input fields during processing
- Display a cancel button

```tsx
export default function App() {
  const { isLoading, stop } = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <form>
      {isLoading && (
        <button key="stop" type="button" onClick={() => stop()}>
          Stop
        </button>
      )}
    </form>
  );
}
```

### Thread Management

Keep track of conversations with built-in thread management. You can access the current thread ID and get notified when new threads are created:

```tsx
const [threadId, setThreadId] = useState<string | null>(null);

const thread = useStream<{ messages: Message[] }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",

  threadId: threadId,
  onThreadId: setThreadId,
});
```

We recommend storing the `threadId` in your URL's query parameters to let users resume conversations after page refreshes.

### Messages Handling

The `useStream()` hook will keep track of the message chunks received from the server and concatenate them together to form a complete message. The completed message chunks can be retrieved via the `messages` property.

By default, the `messagesKey` is set to `messages`, where it will append the new messages chunks to `values["messages"]`. If you store messages in a different key, you can change the value of `messagesKey`.

```tsx
import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";

export default function HomePage() {
  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      {thread.messages.map((message) => (
        <div key={message.id}>{message.content as string}</div>
      ))}
    </div>
  );
}
```

Under the hood, the `useStream()` hook will use the `streamMode: "messages-key"` to receive a stream of messages (i.e. individual LLM tokens) from any LangChain chat model invocations inside your graph nodes. Learn more about messages streaming in the [How to stream messages from your graph](./stream_messages.md) guide.

### Interrupts

The `useStream()` hook exposes the `interrupt` property, which will be filled with the last interrupt from the thread. You can use interrupts to:

- Render a confirmation UI before executing a node
- Wait for human input, allowing agent to ask the user with clarifying questions

Learn more about interrupts in the [How to handle interrupts](../../how-tos/human_in_the_loop/wait-user-input.ipynb) guide.

```tsx
const thread = useStream<{ messages: Message[] }, { InterruptType: string }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});

if (thread.interrupt) {
  return (
    <div>
      Interrupted! {thread.interrupt.value}
      <button
        type="button"
        onClick={() => {
          // `resume` can be any value that the agent accepts
          thread.submit(undefined, { command: { resume: true } });
        }}
      >
        Resume
      </button>
    </div>
  );
}
```

### Branching

For each message, you can use `getMessagesMetadata()` to get the first checkpoint from which the message has been first seen. You can then create a new run from the checkpoint preceding the first seen checkpoint to create a new branch in a thread.

A branch can be created in following ways:

1. Edit a previous user message.
2. Request a regeneration of a previous assistant message.

```tsx
"use client";

import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";
import { useState } from "react";

function BranchSwitcher({
  branch,
  branchOptions,
  onSelect,
}: {
  branch: string | undefined;
  branchOptions: string[] | undefined;
  onSelect: (branch: string) => void;
}) {
  if (!branchOptions || !branch) return null;
  const index = branchOptions.indexOf(branch);

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => {
          const prevBranch = branchOptions[index - 1];
          if (!prevBranch) return;
          onSelect(prevBranch);
        }}
      >
        Prev
      </button>
      <span>
        {index + 1} / {branchOptions.length}
      </span>
      <button
        type="button"
        onClick={() => {
          const nextBranch = branchOptions[index + 1];
          if (!nextBranch) return;
          onSelect(nextBranch);
        }}
      >
        Next
      </button>
    </div>
  );
}

function EditMessage({
  message,
  onEdit,
}: {
  message: Message;
  onEdit: (message: Message) => void;
}) {
  const [editing, setEditing] = useState(false);

  if (!editing) {
    return (
      <button type="button" onClick={() => setEditing(true)}>
        Edit
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const form = e.target as HTMLFormElement;
        const content = new FormData(form).get("content") as string;

        form.reset();
        onEdit({ type: "human", content });
        setEditing(false);
      }}
    >
      <input name="content" defaultValue={message.content as string} />
      <button type="submit">Save</button>
    </form>
  );
}

export default function App() {
  const thread = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      <div>
        {thread.messages.map((message) => {
          const meta = thread.getMessagesMetadata(message);
          const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;

          return (
            <div key={message.id}>
              <div>{message.content as string}</div>

              {message.type === "human" && (
                <EditMessage
                  message={message}
                  onEdit={(message) =>
                    thread.submit(
                      { messages: [message] },
                      { checkpoint: parentCheckpoint }
                    )
                  }
                />
              )}

              {message.type === "ai" && (
                <button
                  type="button"
                  onClick={() =>
                    thread.submit(undefined, { checkpoint: parentCheckpoint })
                  }
                >
                  <span>Regenerate</span>
                </button>
              )}

              <BranchSwitcher
                branch={meta?.branch}
                branchOptions={meta?.branchOptions}
                onSelect={(branch) => thread.setBranch(branch)}
              />
            </div>
          );
        })}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();

          const form = e.target as HTMLFormElement;
          const message = new FormData(form).get("message") as string;

          form.reset();
          thread.submit({ messages: [message] });
        }}
      >
        <input type="text" name="message" />

        {thread.isLoading ? (
          <button key="stop" type="button" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button key="submit" type="submit">
            Send
          </button>
        )}
      </form>
    </div>
  );
}
```

For advanced use cases you can use the `experimental_branchTree` property to get the tree representation of the thread, which can be used to render branching controls for non-message based graphs.

### Optimistic Updates

You can optimistically update the client state before performing a network request to the agent, allowing you to provide immediate feedback to the user, such as showing the user message immediately before the agent has seen the request.

```tsx
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});

const handleSubmit = (text: string) => {
  const newMessage = { type: "human" as const, content: text };

  stream.submit(
    { messages: [newMessage] },
    {
      optimisticValues(prev) {
        const prevMessages = prev.messages ?? [];
        const newMessages = [...prevMessages, newMessage];
        return { ...prev, messages: newMessages };
      },
    }
  );
};
```

### TypeScript

The `useStream()` hook is friendly for apps written in TypeScript and you can specify types for the state to get better type safety and IDE support.

```tsx
// Define your types
type State = {
  messages: Message[];
  context?: Record<string, unknown>;
};

// Use them with the hook
const thread = useStream<State>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

You can also optionally specify types for different scenarios, such as:

- `ConfigurableType`: Type for the `config.configurable` property (default: `Record<string, unknown>`)
- `InterruptType`: Type for the interrupt value - i.e. contents of `interrupt(...)` function (default: `unknown`)
- `CustomEventType`: Type for the custom events (default: `unknown`)
- `UpdateType`: Type for the submit function (default: `Partial<State>`)

```tsx
const thread = useStream<
  State,
  {
    UpdateType: {
      messages: Message[] | Message;
      context?: Record<string, unknown>;
    };
    InterruptType: string;
    CustomEventType: {
      type: "progress" | "debug";
      payload: unknown;
    };
    ConfigurableType: {
      model: string;
    };
  }
>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

If you're using LangGraph.js, you can also reuse your graph's annotation types. However, make sure to only import the types of the annotation schema in order to avoid importing the entire LangGraph.js runtime (i.e. via `import type { ... }` directive).

```tsx
import {
  Annotation,
  MessagesAnnotation,
  type StateType,
  type UpdateType,
} from "@langchain/langgraph/web";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  context: Annotation<string>(),
});

const thread = useStream<
  StateType<typeof AgentState.spec>,
  { UpdateType: UpdateType<typeof AgentState.spec> }
>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

## Event Handling

The `useStream()` hook provides several callback options to help you respond to different events:

- `onError`: Called when an error occurs.
- `onFinish`: Called when the stream is finished.
- `onUpdateEvent`: Called when an update event is received.
- `onCustomEvent`: Called when a custom event is received. See [Custom events](../../concepts/streaming.md#custom) to learn how to stream custom events.
- `onMetadataEvent`: Called when a metadata event is received, which contains the Run ID and Thread ID.

## Learn More

- [JS/TS SDK Reference](../reference/sdk/js_ts_sdk_ref.md)
