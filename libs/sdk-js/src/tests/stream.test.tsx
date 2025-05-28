import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { useStream } from "../react/stream.js";
import "@testing-library/jest-dom/vitest";

function TestChatComponent() {
  const { messages, isLoading, error, submit, stop } = useStream({
    assistantId: "test-assistant",
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

// Mock server setup

const server = setupServer(
  // Mock thread creation
  http.post("*/threads", () => {
    return HttpResponse.json({ thread_id: "test-thread-id" });
  }),

  // Mock stream endpoint
  http.post("*/threads/:threadId/runs/stream", async () => {
    const encoder = new TextEncoder();
    const sendSSE = (event: string, data: unknown) =>
      encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    const stream = new ReadableStream({
      async start(controller) {
        await new Promise((resolve) => setTimeout(resolve, 10));

        controller.enqueue(
          sendSSE("metadata", {
            run_id: "1f03278a-1734-6518-80a4-3390db59f960",
            attempt: 1,
          }),
        );

        controller.enqueue(
          sendSSE("values", {
            messages: [
              {
                content: "Hey",
                additional_kwargs: {},
                response_metadata: {},
                type: "human",
                name: null,
                id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                example: false,
              },
            ],
          }),
        );

        controller.enqueue(
          sendSSE("messages", [
            {
              content: "",
              additional_kwargs: {},
              response_metadata: { model_name: "claude-3-7-sonnet-latest" },
              type: "AIMessageChunk",
              name: null,
              id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
              tool_calls: [],
              invalid_tool_calls: [],
              tool_call_chunks: [],
            },
            { run_attempt: 1 },
          ]),
        );

        controller.enqueue(
          sendSSE("messages", [
            {
              content: "Hello",
              additional_kwargs: {},
              response_metadata: { model_name: "claude-3-7-sonnet-latest" },
              type: "AIMessageChunk",
              name: null,
              id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
              tool_calls: [],
              invalid_tool_calls: [],
              tool_call_chunks: [],
            },
            { run_attempt: 1 },
          ]),
        );

        controller.enqueue(
          sendSSE("messages", [
            {
              content: "! How can I assist you today?",
              additional_kwargs: {},
              response_metadata: { model_name: "claude-3-7-sonnet-latest" },
              type: "AIMessageChunk",
              name: null,
              id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
              tool_calls: [],
              invalid_tool_calls: [],
              tool_call_chunks: [],
            },
            { run_attempt: 1 },
          ]),
        );

        controller.enqueue(
          sendSSE("messages", [
            {
              content: "",
              additional_kwargs: {},
              response_metadata: {
                stop_reason: "end_turn",
                stop_sequence: null,
              },
              type: "AIMessageChunk",
              name: null,
              id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
              tool_calls: [],
              invalid_tool_calls: [],
              tool_call_chunks: [],
            },
            { run_attempt: 1 },
          ]),
        );

        controller.enqueue(
          sendSSE("values", {
            messages: [
              {
                content: "Hey",
                additional_kwargs: {},
                response_metadata: {},
                type: "human",
                name: null,
                id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                example: false,
              },
              {
                content: "Hello! How can I assist you today?",
                additional_kwargs: {},
                response_metadata: {
                  model_name: "claude-3-7-sonnet-latest",
                  stop_reason: "end_turn",
                  stop_sequence: null,
                },
                type: "ai",
                name: null,
                id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
                tool_calls: [],
                invalid_tool_calls: [],
              },
            ],
          }),
        );

        controller.close();
      },
    });

    server.use(
      http.post("*/threads/:threadId/history", () => {
        return HttpResponse.json([
          {
            values: {
              messages: [
                {
                  content: "Hey",
                  additional_kwargs: {},
                  response_metadata: {},
                  type: "human",
                  name: null,
                  id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                  example: false,
                },
                {
                  content: "Hello! How can I assist you today?",
                  additional_kwargs: {},
                  response_metadata: {
                    model_name: "claude-3-7-sonnet-latest",
                    stop_reason: "end_turn",
                    stop_sequence: null,
                  },
                  type: "ai",
                  name: null,
                  id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
                  example: false,
                  tool_calls: [],
                  invalid_tool_calls: [],
                },
              ],
            },
            next: [],
            tasks: [],
            metadata: {
              run_attempt: 1,
              source: "loop",
              writes: {
                agent: {
                  messages: [
                    {
                      content: "Hello! How can I assist you today?",
                      additional_kwargs: {},
                      response_metadata: {
                        model_name: "claude-3-7-sonnet-latest",
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      type: "ai",
                      name: null,
                      id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
                      example: false,
                      tool_calls: [],
                      invalid_tool_calls: [],
                    },
                  ],
                },
              },
              step: 1,
              parents: {},
            },
            created_at: "2025-05-16T17:10:16.987537+00:00",
            checkpoint: {
              checkpoint_id: "1f03278a-38cf-6c68-8001-22b77ac43ff6",
              thread_id: "b06fd92a-955c-446e-b233-7977716c4a9c",
              checkpoint_ns: "",
            },
            parent_checkpoint: {
              checkpoint_id: "1f03278a-206b-67c6-8000-ac34a0872e1a",
              thread_id: "b06fd92a-955c-446e-b233-7977716c4a9c",
              checkpoint_ns: "",
            },
            checkpoint_id: "1f03278a-38cf-6c68-8001-22b77ac43ff6",
            parent_checkpoint_id: "1f03278a-206b-67c6-8000-ac34a0872e1a",
          },
          {
            values: {
              messages: [
                {
                  content: "Hey",
                  additional_kwargs: {},
                  response_metadata: {},
                  type: "human",
                  name: null,
                  id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                  example: false,
                },
              ],
            },
            next: ["agent"],
            tasks: [
              {
                id: "e1b7b52b-a78e-4b32-0c89-e06bf46405ed",
                name: "agent",
                path: ["__pregel_pull", "agent"],
                error: null,
                interrupts: [],
                checkpoint: null,
                state: null,
                result: {
                  messages: [
                    {
                      content: "Hello! How can I assist you today?",
                      additional_kwargs: {},
                      response_metadata: {
                        model_name: "claude-3-7-sonnet-latest",
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      type: "ai",
                      name: null,
                      id: "run-3e90ba6a-71d6-49e7-94a8-6bcac2fd0f40",
                      example: false,
                      tool_calls: [],
                      invalid_tool_calls: [],
                    },
                  ],
                },
              },
            ],
            metadata: {
              run_attempt: 1,
            },
            created_at: "2025-05-16T17:10:14.429889+00:00",
            checkpoint: {
              checkpoint_id: "1f03278a-206b-67c6-8000-ac34a0872e1a",
              thread_id: "b06fd92a-955c-446e-b233-7977716c4a9c",
              checkpoint_ns: "",
            },
            parent_checkpoint: {
              checkpoint_id: "1f03278a-2067-6590-bfff-3fb740466fc3",
              thread_id: "b06fd92a-955c-446e-b233-7977716c4a9c",
              checkpoint_ns: "",
            },
            checkpoint_id: "1f03278a-206b-67c6-8000-ac34a0872e1a",
            parent_checkpoint_id: "1f03278a-2067-6590-bfff-3fb740466fc3",
          },
          {
            values: {
              messages: [],
            },
            next: ["__start__"],
            tasks: [
              {
                id: "291af033-2ddc-3320-8bbc-28060057cae5",
                name: "__start__",
                path: ["__pregel_pull", "__start__"],
                error: null,
                interrupts: [],
                checkpoint: null,
                state: null,
                result: {
                  messages: [
                    {
                      id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                      type: "human",
                      content: "Hey",
                    },
                  ],
                },
              },
            ],
            metadata: {
              run_attempt: 1,
              source: "input",
              writes: {
                __start__: {
                  messages: [
                    {
                      id: "2d8c0d9f-a614-4e44-b474-6a56e9471cf5",
                      type: "human",
                      content: "Hey",
                    },
                  ],
                },
              },
              step: -1,
              parents: {},
            },
            created_at: "2025-05-16T17:10:14.428191+00:00",
            checkpoint: {
              checkpoint_id: "1f03278a-2067-6590-bfff-3fb740466fc3",
              thread_id: "b06fd92a-955c-446e-b233-7977716c4a9c",
              checkpoint_ns: "",
            },
            parent_checkpoint: null,
            checkpoint_id: "1f03278a-2067-6590-bfff-3fb740466fc3",
            parent_checkpoint_id: null,
          },
        ]);
      }),
    );

    return new HttpResponse(stream, {
      headers: { "Content-Type": "text/event-stream" },
    });
  }),
);

server.use;

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
      expect(screen.getByTestId("message-0")).toHaveTextContent("Hey");
      expect(screen.getByTestId("message-1")).toHaveTextContent(
        "Hello! How can I assist you today?",
      );
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
