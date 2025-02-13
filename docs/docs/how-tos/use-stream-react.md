# How to stream runs into an React app

!!! info "Prerequisites" - [LangGraph Platform](../concepts/langgraph_platform.md) - [LangGraph Server](../concepts/langgraph_server.md)

The `useStream()` hook allows you to easily stream values from a LangGraph run. It enables the following features:

- Streaming messages: Streams messages from the run as they are generated.
- State management: Thread state is managed for you, including messages, loading and error states.
- Branching support: We handle checkpoint branching for you, so you can focus on building your chat interface.
- Headless: Bring your own chat UI and implement streaming into any design or layout.

This guide will show you how you can use `useStream()` to stream values within your React application.

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
          <button key="submit" type="submit">
            Send
          </button>
        )}
      </form>
    </div>
  );
}
```

## Customise UI

The `useStream()` hook provides built-in state management capabilities to simplify your application development. It handles:

- Thread state management
- Loading states during stream operations
- Error handling and error states
- Message management

This allows you to focus on building your UI while the `useStream()` hook takes care of the underlying state complexity.

### Loading state

The `isLoading` property is set to `true` whenever the stream is running. This is useful for:

1. Showing a loading spinner to indicate that the stream is running.
2. Disabling the input box to prevent multiple submissions.
3. Showing a cancellation button to cancel a run.

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

### Thread management

The `useStream()` hook manages a thread for you. You can use the `threadId` property to get the thread ID. Pass in the `onThreadId` callback to get notified when the new thread is created.

```tsx
const [threadId, setThreadId] = useState<string | null>(null);

const thread = useStream<{ messages: Message[] }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",

  threadId: threadId,
  onThreadId: setThreadId,
});
```

We recommend setting the `threadId` as a query parameter in the URL, so that you can resume the conversation from the same thread even when the page is refreshed.

### Messages handling

To enable messages handling, you need to pass the `messagesKey` option to the `useStream()` hook. When enabled, the `useStream()` hook will keep track of the message chunks received from the server and concatenate them together to form a complete message. The completed message chunks can be retrieved via the `messages` property.

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

### Branching

To enable branching, you need to enable messages handling. Pass the `messagesKey` option to the `useStream()` hook. For each message, you can use `getMessagesMetadata()` to get the first checkpoint from which the message has been first seen. You can then create a new run from the checkpoint preceding the first seen checkpoint to create a new branch in a thread.

A branch can be created in following ways:

1. Edit a previous user message.
2. Request a regeneration of a previous assistant message.

```tsx
/* eslint-disable @typescript-eslint/no-floating-promises */
"use client";

import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";
import {
  Annotation,
  MessagesAnnotation,
  type StateType,
  type UpdateType,
} from "@langchain/langgraph/web";
import { useState } from "react";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
});

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
  const thread = useStream<
    StateType<typeof AgentState.spec>,
    UpdateType<typeof AgentState.spec>
  >({
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
                      { checkpoint: parentCheckpoint },
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

### TypeScript and Type safety

The `useStream()` hook accepts generic parameters that can be used to specify the thread state and update type as well as the custom event type, avoiding the need to manually type-cast.

```tsx
// Type definition of the state
type StateType = { messages: Message[] };

// Type definition of the update
type UpdateType = { messages: Message[] | Message };

// Type definition of the custom event
type CustomEventType = { counter: number };

const thread = useStream<StateType, UpdateType, CustomEventType>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

If you use `LangGraph.js`, you can re-use the same `Annotation` as the one used within `StateGraph`.

!!! warning "Importing from @langchain/langgraph/web"

    Make sure to import from `@langchain/langgraph/web` and not from `@langchain/langgraph`, as the default entrypoint will attempt to initialize `AsyncLocalStorage`, which is not available in the browser.

```tsx
"use client";

import { useStream } from "@langchain/langgraph-sdk/react";
import {
  Annotation,
  MessagesAnnotation,
  type StateType,
  type UpdateType,
} from "@langchain/langgraph/web";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
});

export default function HomePage() {
  const thread = useStream<
    StateType<typeof AgentState.spec>,
    UpdateType<typeof AgentState.spec>
  >({
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

## Event callbacks

The `useStream()` hook provides few event callbacks that you can use to react to specific events.

- `onError`: Called when an error occurs.
- `onFinish`: Called when the stream is finished.
- `onUpdateEvent`: Called when an update event is received.
- `onCustomEvent`: Called when a custom event is received. See [Custom events](../concepts/custom-events.md) to learn how to stream custom events.
- `onMetadataEvent`: Called when a metadata event is received.

## Learn more

TODO: add a link to the `useStream()` hook documentation.
