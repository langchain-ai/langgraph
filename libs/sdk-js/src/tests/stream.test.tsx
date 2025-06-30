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
      <button
        data-testid="submit"
        onClick={() =>
          submit({ messages: [{ content: "Hello", type: "human" }] })
        }
      >
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
});
