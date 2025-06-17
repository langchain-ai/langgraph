import "@testing-library/jest-dom/vitest";

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { setupServer } from "msw/node";
import { http } from "msw";
import { useStream } from "../react/stream.js";

import { StateGraph, MessagesAnnotation, START } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { FakeStreamingChatModel } from "@langchain/core/utils/testing";
import { AIMessage, BaseMessageLike } from "@langchain/core/messages";

import { Hono } from "hono";
import { logger } from "hono/logger";
import { createEmbedServer } from "@langchain/langgraph-api/experimental/embed";

const threads = (() => {
  const THREADS: Record<
    string,
    { thread_id: string; metadata: Record<string, unknown> }
  > = {};

  return {
    get: async (id: string) => THREADS[id],
    put: async (
      threadId: string,
      { metadata }: { metadata?: Record<string, unknown> },
    ) => {
      THREADS[threadId] = { thread_id: threadId, metadata: metadata ?? {} };
    },
    delete: async (threadId: string) => {
      delete THREADS[threadId];
    },
  };
})();

const checkpointer = new MemorySaver();

const model = new FakeStreamingChatModel({ responses: [new AIMessage("Hey")] });
const agent = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state: { messages: BaseMessageLike[] }) => {
    const response = await model.invoke(state.messages);
    return { messages: [response] };
  })
  .addEdge(START, "agent")
  .compile();

const app = new Hono();
app.use(logger());
app.route("/", createEmbedServer({ graph: { agent }, checkpointer, threads }));

const server = setupServer(http.all("*", (ctx) => app.fetch(ctx.request)));

function TestChatComponent() {
  const { messages, isLoading, error, submit, stop } = useStream({
    assistantId: "agent",
    apiKey: "test-api-key",
  });

  return (
    <div>
      <div data-testid="messages">
        {messages.map((msg, i) => (
          <div key={msg.id ?? i} data-testid={`message-${i}`}>
            {typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content)}
          </div>
        ))}
      </div>
      <div data-testid="loading">
        {isLoading ? "Loading..." : "Not loading"}
      </div>
      {error ? <div data-testid="error">{String(error)}</div> : null}
      <button data-testid="submit" onClick={() => submit({})}>
        Send
      </button>
      <button data-testid="stop" onClick={stop}>
        Stop
      </button>
    </div>
  );
}

describe("useStream", () => {
  beforeEach(() => server.listen());

  afterEach(() => {
    server.resetHandlers();
    server.close();
    vi.clearAllMocks();
  });

  it("renders initial state correctly", () => {
    render(<TestChatComponent />);

    expect(screen.getByTestId("loading")).toHaveTextContent("Not loading");
    expect(screen.getByTestId("messages")).toBeEmptyDOMElement();
    expect(screen.queryByTestId("error")).not.toBeInTheDocument();
  });

  it("handles message submission and streaming", async () => {
    const user = userEvent.setup();

    render(<TestChatComponent />);

    // Check loading state
    await user.click(screen.getByTestId("submit"));
    expect(screen.getByTestId("loading")).toHaveTextContent("Loading...");

    // Wait for messages to appear
    await waitFor(() => {
      expect(screen.getByTestId("message-0")).toHaveTextContent("Hello");
      expect(screen.getByTestId("message-1")).toHaveTextContent("Hey");
    });

    // Check final state
    expect(screen.getByTestId("loading")).toHaveTextContent("Not loading");
  });

  it("handles stop functionality", async () => {
    const user = userEvent.setup();
    render(<TestChatComponent />);

    // Start streaming and stop immediately
    await user.click(screen.getByTestId("submit"));
    await user.click(screen.getByTestId("stop"));

    // Check loading state is reset
    await waitFor(() => {
      expect(screen.getByTestId("loading")).toHaveTextContent("Not loading");
    });
  });

  it("displays initial values immediately", () => {
    const initialValues = {
      messages: [
        { id: "cached-1", type: "human", content: "Cached user message" },
        { id: "cached-2", type: "ai", content: "Cached AI response" },
      ],
    };

    function TestCachedComponent() {
      const { messages, values } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        initialValues,
      });

      return (
        <div>
          <div data-testid="messages">
            {messages.map((msg, i) => (
              <div key={msg.id ?? i} data-testid={`message-${i}`}>
                {typeof msg.content === "string"
                  ? msg.content
                  : JSON.stringify(msg.content)}
              </div>
            ))}
          </div>
          <div data-testid="values">{JSON.stringify(values)}</div>
        </div>
      );
    }

    render(<TestCachedComponent />);

    // Should immediately show cached messages
    expect(screen.getByTestId("message-0")).toHaveTextContent(
      "Cached user message",
    );
    expect(screen.getByTestId("message-1")).toHaveTextContent(
      "Cached AI response",
    );

    // Values should include initial values
    expect(screen.getByTestId("values")).toHaveTextContent(
      "Cached user message",
    );
  });

  it("handles null initial values", () => {
    function TestNullInitialComponent() {
      const { messages, values } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        initialValues: null,
      });

      return (
        <div>
          <div data-testid="message-count">{messages.length}</div>
          <div data-testid="values">{JSON.stringify(values)}</div>
        </div>
      );
    }

    render(<TestNullInitialComponent />);

    // Should handle null initialValues gracefully
    expect(screen.getByTestId("message-count")).toHaveTextContent("0");
    expect(screen.getByTestId("values")).toHaveTextContent("{}");
  });

  it("accepts newThreadId option without errors", () => {
    // Test that newThreadId option can be passed without causing errors
    function TestNewThreadComponent() {
      const stream = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        threadId: null, // Start with no thread
        newThreadId: "predetermined-thread-id",
        onThreadId: () => {}, // Mock callback
      });

      return (
        <div>
          <div data-testid="loading">
            {stream.isLoading ? "Loading..." : "Not loading"}
          </div>
          <div data-testid="thread-id">
            {stream.client ? "Client ready" : "No client"}
          </div>
        </div>
      );
    }

    render(<TestNewThreadComponent />);

    // Should render without errors
    expect(screen.getByTestId("loading")).toHaveTextContent("Not loading");
    expect(screen.getByTestId("thread-id")).toHaveTextContent("Client ready");
  });

  it("shows initial values before any streaming", () => {
    const initialValues = {
      messages: [
        { id: "initial-1", type: "human", content: "Initial message" },
      ],
      customField: "initial-value",
    };

    function TestInitialValuesComponent() {
      const stream = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        initialValues,
      });

      return (
        <div>
          <div data-testid="message-0">
            {typeof stream.messages[0]?.content === "string"
              ? stream.messages[0]?.content
              : JSON.stringify(stream.messages[0]?.content)}
          </div>
          <div data-testid="custom-field">{stream.values.customField}</div>
          <div data-testid="loading">
            {stream.isLoading ? "Loading" : "Not loading"}
          </div>
        </div>
      );
    }

    render(<TestInitialValuesComponent />);

    // Should immediately show initial values without any loading
    expect(screen.getByTestId("message-0")).toHaveTextContent(
      "Initial message",
    );
    expect(screen.getByTestId("custom-field")).toHaveTextContent(
      "initial-value",
    );
    expect(screen.getByTestId("loading")).toHaveTextContent("Not loading");
  });
});

describe("useStream onStop callback", () => {
  const user = userEvent.setup();

  it("calls onStop callback when stop is called", async () => {
    const onStopCallback = vi.fn();

    function TestComponent() {
      const { submit, stop } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        onStop: onStopCallback,
      });

      return (
        <div>
          <button data-testid="submit" onClick={() => submit({})}>
            Send
          </button>
          <button data-testid="stop" onClick={stop}>
            Stop
          </button>
        </div>
      );
    }

    render(<TestComponent />);

    // Start a stream and stop it
    await user.click(screen.getByTestId("submit"));
    await user.click(screen.getByTestId("stop"));

    // Verify onStop was called with mutate function
    expect(onStopCallback).toHaveBeenCalledTimes(1);
    expect(onStopCallback).toHaveBeenCalledWith(
      expect.objectContaining({
        mutate: expect.any(Function),
      }),
    );
  });

  it("mutate function updates stream values immediately", async () => {
    function TestComponent() {
      const { submit, stop, values } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        onStop: ({ mutate }) => {
          mutate((prev) => ({
            ...prev,
            stoppedByUser: true,
            customMessage: "Stream stopped",
          }));
        },
      });

      return (
        <div>
          <div data-testid="stopped-status">
            {(values as any).stoppedByUser ? "Stopped" : "Not stopped"}
          </div>
          <div data-testid="custom-message">
            {(values as any).customMessage || "No message"}
          </div>
          <button data-testid="submit" onClick={() => submit({})}>
            Send
          </button>
          <button data-testid="stop" onClick={stop}>
            Stop
          </button>
        </div>
      );
    }

    render(<TestComponent />);

    // Initial state
    expect(screen.getByTestId("stopped-status")).toHaveTextContent(
      "Not stopped",
    );
    expect(screen.getByTestId("custom-message")).toHaveTextContent(
      "No message",
    );

    // Start and stop stream
    await user.click(screen.getByTestId("submit"));
    await user.click(screen.getByTestId("stop"));

    // Verify state was updated immediately
    await waitFor(() => {
      expect(screen.getByTestId("stopped-status")).toHaveTextContent("Stopped");
      expect(screen.getByTestId("custom-message")).toHaveTextContent(
        "Stream stopped",
      );
    });
  });

  it("handles functional updates correctly", async () => {
    function TestComponent() {
      const { submit, stop, values } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        initialValues: {
          counter: 5,
          items: ["item1", "item2"],
        },
        onStop: ({ mutate }) => {
          mutate((prev: any) => ({
            ...prev,
            counter: (prev.counter || 0) + 10,
            items: [...(prev.items || []), "stopped"],
          }));
        },
      });

      return (
        <div>
          <div data-testid="counter">{(values as any).counter}</div>
          <div data-testid="items">{(values as any).items?.join(", ")}</div>
          <button data-testid="submit" onClick={() => submit({})}>
            Send
          </button>
          <button data-testid="stop" onClick={stop}>
            Stop
          </button>
        </div>
      );
    }

    render(<TestComponent />);

    // Initial state
    expect(screen.getByTestId("counter")).toHaveTextContent("5");
    expect(screen.getByTestId("items")).toHaveTextContent("item1, item2");

    // Start and stop stream
    await user.click(screen.getByTestId("submit"));
    await user.click(screen.getByTestId("stop"));

    // Verify functional update was applied correctly
    await waitFor(() => {
      expect(screen.getByTestId("counter")).toHaveTextContent("15");
      expect(screen.getByTestId("items")).toHaveTextContent(
        "item1, item2, stopped",
      );
    });
  });

  it("is not called when stream completes naturally", async () => {
    const onStopCallback = vi.fn();

    function TestComponent() {
      const { submit } = useStream({
        assistantId: "test-assistant",
        apiKey: "test-api-key",
        onStop: onStopCallback,
      });

      return (
        <div>
          <button data-testid="submit" onClick={() => submit({})}>
            Send
          </button>
        </div>
      );
    }

    render(<TestComponent />);

    // Start a stream and let it complete naturally
    await user.click(screen.getByTestId("submit"));

    // Wait for stream to complete naturally
    await waitFor(() => {
      expect(onStopCallback).not.toHaveBeenCalled();
    });
  });
});
