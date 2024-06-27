import {
  Assistant,
  AssistantGraph,
  Config,
  DefaultValues,
  GraphSchema,
  Metadata,
  Run,
  Thread,
  ThreadState,
  Cron,
} from "./schema.js";
import { AsyncCaller, AsyncCallerParams } from "./utils/async_caller.mjs";
import { EventSourceParser, createParser } from "eventsource-parser";
import { IterableReadableStream } from "./utils/stream.mjs";
import {
  RunsCreatePayload,
  RunsStreamPayload,
  RunsWaitPayload,
  StreamEvent,
  CronsCreatePayload,
} from "./types.mjs";

interface ClientConfig {
  apiUrl?: string;
  callerOptions?: AsyncCallerParams;
  timeoutMs?: number;
  defaultHeaders?: Record<string, string | null | undefined>;
}

class BaseClient {
  protected asyncCaller: AsyncCaller;

  protected timeoutMs: number;

  protected apiUrl: string;

  protected defaultHeaders: Record<string, string | null | undefined>;

  constructor(config?: ClientConfig) {
    this.asyncCaller = new AsyncCaller({
      maxRetries: 4,
      maxConcurrency: 4,
      ...config?.callerOptions,
    });

    this.timeoutMs = config?.timeoutMs || 12_000;
    this.apiUrl = config?.apiUrl || "http://localhost:8123";
    this.defaultHeaders = config?.defaultHeaders || {};
  }

  protected prepareFetchOptions(
    path: string,
    options?: RequestInit & {
      json?: unknown;
      params?: Record<string, unknown>;
    },
  ): [url: URL, init: RequestInit] {
    const mutatedOptions = {
      ...options,
      headers: { ...this.defaultHeaders, ...options?.headers },
    };

    if (mutatedOptions.json) {
      mutatedOptions.body = JSON.stringify(mutatedOptions.json);
      mutatedOptions.headers = {
        ...mutatedOptions.headers,
        "Content-Type": "application/json",
      };
      delete mutatedOptions.json;
    }

    const targetUrl = new URL(`${this.apiUrl}${path}`);

    if (mutatedOptions.params) {
      for (const [key, value] of Object.entries(mutatedOptions.params)) {
        if (value == null) continue;

        let strValue =
          typeof value === "string" || typeof value === "number"
            ? value.toString()
            : JSON.stringify(value);

        targetUrl.searchParams.append(key, strValue);
      }
      delete mutatedOptions.params;
    }

    return [targetUrl, mutatedOptions];
  }

  protected async fetch<T>(
    path: string,
    options?: RequestInit & {
      json?: unknown;
      params?: Record<string, unknown>;
    },
  ): Promise<T> {
    const response = await this.asyncCaller.fetch(
      ...this.prepareFetchOptions(path, options),
    );
    if (response.status === 202 || response.status === 204) {
      return undefined as T;
    }
    return response.json() as T;
  }
}

class CronsClient extends BaseClient {
  /**
   * 
   * @param threadId The ID of the thread.
   * @param assistantId Assistant ID to use for this cron job.
   * @param payload Payload for creating a cron job.
   * @returns The created background run.
   */
  async create_for_thread(
    threadId: string,
    assistantId: string,
    payload?: CronsCreatePayload,
  ): Promise<Run> {
    const json: Record<string, any> = {
      schedule: payload?.schedule,
      input: payload?.input,
      config: payload?.config,
      metadata: payload?.metadata,
      assistant_id: assistantId,
      interrupt_before: payload?.interruptBefore,
      interrupt_after: payload?.interruptAfter,
      webhook: payload?.webhook,
    };
    return this.fetch<Run>(`/threads/${threadId}/runs/crons`, {
      method: "POST",
      json,
      signal: payload?.signal,
    });
  }

  /**
   * 
   * @param assistantId Assistant ID to use for this cron job.
   * @param payload Payload for creating a cron job.
   * @returns 
   */
  async create(
    assistantId: string,
    payload?: CronsCreatePayload,
  ): Promise<Run> {
    const json: Record<string, any> = {
      schedule: payload?.schedule,
      input: payload?.input,
      config: payload?.config,
      metadata: payload?.metadata,
      assistant_id: assistantId,
      interrupt_before: payload?.interruptBefore,
      interrupt_after: payload?.interruptAfter,
      webhook: payload?.webhook,
    };
    return this.fetch<Run>(`/runs/crons`, {
      method: "POST",
      json,
      signal: payload?.signal,
    });
  }

  /**
   * 
   * @param cronId Cron ID of Cron job to delete.
   */
  async delete(cronId: string): Promise<void> {
      await this.fetch<void>(`/runs/crons/${cronId}`, {
        method: "DELETE",
      });
  }

  /**
   * 
   * @param query Query options.
   * @returns List of crons.
   */
  async search(query?: {
    assistantId?: string;
    threadId?: string;
    limit?: number;
    offset?: number;
  }): Promise<Cron[]> {
    return this.fetch<Cron[]>("/runs/crons/search", {
      method: "POST",
      json: {
        assistant_id: query?.assistantId ?? undefined,
        thread_id: query?.threadId ?? undefined,
        limit: query?.limit ?? 10,
        offset: query?.offset ?? 0,
      }
    })
  }

}

class AssistantsClient extends BaseClient {
  /**
   * Get an assistant by ID.
   *
   * @param assistantId The ID of the assistant.
   * @returns Assistant
   */
  async get(assistantId: string): Promise<Assistant> {
    return this.fetch<Assistant>(`/assistants/${assistantId}`);
  }

  /**
   * Get the JSON representation of the graph assigned to a runnable
   * @param assistantId The ID of the assistant.
   * @returns Serialized graph
   */
  async getGraph(assistantId: string): Promise<AssistantGraph> {
    return this.fetch<AssistantGraph>(`/assistants/${assistantId}/graph`);
  }

  /**
   * Get the state and config schema of the graph assigned to a runnable
   * @param assistantId The ID of the assistant.
   * @returns Graph schema
   */
  async getSchemas(assistantId: string): Promise<GraphSchema> {
    return this.fetch<GraphSchema>(`/assistants/${assistantId}/schemas`);
  }

  /**
   * Create a new assistant.
   * @param payload Payload for creating an assistant.
   * @returns The created assistant.
   */
  async create(payload: {
    graphId: string;
    config?: Config;
    metadata?: Metadata;
  }): Promise<Assistant> {
    return this.fetch<Assistant>("/assistants", {
      method: "POST",
      json: {
        graph_id: payload.graphId,
        config: payload.config,
        metadata: payload.metadata,
      },
    });
  }

  /**
   * Update an assistant.
   * @param assistantId ID of the assistant.
   * @param payload Payload for updating the assistant.
   * @returns The updated assistant.
   */
  async update(
    assistantId: string,
    payload: {
      graphId: string;
      config?: Config;
      metadata?: Metadata;
    },
  ): Promise<Assistant> {
    return this.fetch<Assistant>(`/assistants/${assistantId}`, {
      method: "PATCH",
      json: {
        graph_id: payload.graphId,
        config: payload.config,
        metadata: payload.metadata,
      },
    });
  }

  /**
   * Delete an assistant.
   *
   * @param assistantId ID of the assistant.
   */
  async delete(assistantId: string): Promise<void> {
    return this.fetch<void>(`/assistants/${assistantId}`, {
      method: "DELETE",
    });
  }

  /**
   * List assistants.
   * @param query Query options.
   * @returns List of assistants.
   */
  async search(query?: {
    metadata?: Metadata;
    limit?: number;
    offset?: number;
  }): Promise<Assistant[]> {
    return this.fetch<Assistant[]>("/assistants/search", {
      method: "POST",
      json: {
        metadata: query?.metadata ?? undefined,
        limit: query?.limit ?? 10,
        offset: query?.offset ?? 0,
      },
    });
  }
}

class ThreadsClient extends BaseClient {
  /**
   * Get a thread by ID.
   *
   * @param threadId ID of the thread.
   * @returns The thread.
   */
  async get(threadId: string): Promise<Thread> {
    return this.fetch<Thread>(`/threads/${threadId}`);
  }

  /**
   * Create a new thread.
   *
   * @param payload Payload for creating a thread.
   * @returns The created thread.
   */
  async create(payload?: {
    /**
     * Metadata for the thread.
     */
    metadata?: Metadata;
  }): Promise<Thread> {
    return this.fetch<Thread>(`/threads`, {
      method: "POST",
      json: { metadata: payload?.metadata },
    });
  }

  /**
   * Update a thread.
   *
   * @param threadId ID of the thread.
   * @param payload Payload for updating the thread.
   * @returns The updated thread.
   */
  async update(
    threadId: string,
    payload?: {
      /**
       * Metadata for the thread.
       */
      metadata?: Metadata;
    },
  ): Promise<Thread> {
    return this.fetch<Thread>(`/threads/${threadId}`, {
      method: "PATCH",
      json: { metadata: payload?.metadata },
    });
  }

  /**
   * Delete a thread.
   *
   * @param threadId ID of the thread.
   */
  async delete(threadId: string): Promise<void> {
    return this.fetch<void>(`/threads/${threadId}`, {
      method: "DELETE",
    });
  }

  /**
   * List threads
   *
   * @param query Query options
   * @returns List of threads
   */
  async search(query?: {
    /**
     * Metadata to filter threads by.
     */
    metadata?: Metadata;
    /**
     * Maximum number of threads to return.
     * Defaults to 10
     */
    limit?: number;
    /**
     * Offset to start from.
     */
    offset?: number;
  }): Promise<Thread[]> {
    return this.fetch<Thread[]>("/threads/search", {
      method: "POST",
      json: {
        metadata: query?.metadata ?? undefined,
        limit: query?.limit ?? 10,
        offset: query?.offset ?? 0,
      },
    });
  }

  /**
   * Get state for a thread.
   *
   * @param threadId ID of the thread.
   * @returns Thread state.
   */
  async getState<ValuesType = DefaultValues>(
    threadId: string,
    checkpointId?: string,
  ): Promise<ThreadState<ValuesType>> {
    return this.fetch<ThreadState<ValuesType>>(
      checkpointId != null
        ? `/threads/${threadId}/state/${checkpointId}`
        : `/threads/${threadId}/state`,
    );
  }

  /**
   * Add state to a thread.
   *
   * @param threadId The ID of the thread.
   * @returns
   */
  async updateState<ValuesType = DefaultValues>(
    threadId: string,
    options: { values: ValuesType; checkpointId?: string; asNode?: string },
  ): Promise<void> {
    return this.fetch<void>(`/threads/${threadId}/state`, {
      method: "POST",
      json: {
        values: options.values,
        checkpoint_id: options.checkpointId,
        as_node: options?.asNode,
      },
    });
  }

  /**
   * Patch the metadata of a thread.
   *
   * @param threadIdOrConfig Thread ID or config to patch the state of.
   * @param metadata Metadata to patch the state with.
   */
  async patchState(
    threadIdOrConfig: string | Config,
    metadata: Metadata,
  ): Promise<void> {
    let threadId: string;

    if (typeof threadIdOrConfig !== "string") {
      if (typeof threadIdOrConfig.configurable.thread_id !== "string") {
        throw new Error(
          "Thread ID is required when updating state with a config.",
        );
      }
      threadId = threadIdOrConfig.configurable.thread_id;
    } else {
      threadId = threadIdOrConfig;
    }

    return this.fetch<void>(`/threads/${threadId}/state`, {
      method: "PATCH",
      json: { metadata: metadata },
    });
  }

  /**
   * Get all past states for a thread.
   *
   * @param threadId ID of the thread.
   * @param options Additional options.
   * @returns List of thread states.
   */
  async getHistory<ValuesType = DefaultValues>(
    threadId: string,
    options?: {
      limit?: number;
      before?: Config;
      metadata?: Metadata;
    },
  ): Promise<ThreadState<ValuesType>[]> {
    return this.fetch<ThreadState<ValuesType>[]>(
      `/threads/${threadId}/history`,
      {
        method: "POST",
        json: {
          limit: options?.limit ?? 10,
          before: options?.before,
          metadata: options?.metadata,
        },
      },
    );
  }
}

class RunsClient extends BaseClient {
  stream(
    threadId: null,
    assistantId: string,
    payload?: Omit<RunsStreamPayload, "multitaskStrategy">,
  ): AsyncGenerator<{
    event: StreamEvent;
    data: any;
  }>;

  stream(
    threadId: string,
    assistantId: string,
    payload?: RunsStreamPayload,
  ): AsyncGenerator<{
    event: StreamEvent;
    data: any;
  }>;

  /**
   * Create a run and stream the results.
   *
   * @param threadId The ID of the thread.
   * @param assistantId Assistant ID to use for this run.
   * @param payload Payload for creating a run.
   */
  async *stream(
    threadId: string | null,
    assistantId: string,
    payload?: RunsStreamPayload,
  ): AsyncGenerator<{
    event: StreamEvent;
    // TODO: figure out a better way to
    // type this without any
    data: any;
  }> {
    const json: Record<string, any> = {
      input: payload?.input,
      config: payload?.config,
      metadata: payload?.metadata,
      stream_mode: payload?.streamMode,
      feedback_keys: payload?.feedbackKeys,
      assistant_id: assistantId,
      interrupt_before: payload?.interruptBefore,
      interrupt_after: payload?.interruptAfter,
    };
    if (payload?.multitaskStrategy != null) {
      json["multitask_strategy"] = payload?.multitaskStrategy;
    }

    const endpoint =
      threadId == null ? `/runs/stream` : `/threads/${threadId}/runs/stream`;
    const response = await this.asyncCaller.fetch(
      ...this.prepareFetchOptions(endpoint, {
        method: "POST",
        json,
        signal: payload?.signal,
      }),
    );

    let parser: EventSourceParser;
    const textDecoder = new TextDecoder();

    const stream: ReadableStream<{ event: string; data: any }> = (
      response.body || new ReadableStream({ start: (ctrl) => ctrl.close() })
    ).pipeThrough(
      new TransformStream({
        async start(ctrl) {
          parser = createParser((event) => {
            if (
              (payload?.signal && payload.signal.aborted) ||
              (event.type === "event" && event.data === "[DONE]")
            ) {
              ctrl.terminate();
              return;
            }

            if ("data" in event) {
              ctrl.enqueue({
                event: event.event ?? "message",
                data: JSON.parse(event.data),
              });
            }
          });
        },
        async transform(chunk) {
          parser.feed(textDecoder.decode(chunk));
        },
      }),
    );

    yield* IterableReadableStream.fromReadableStream(stream);
  }

  /**
   * Create a run.
   *
   * @param threadId The ID of the thread.
   * @param assistantId Assistant ID to use for this run.
   * @param payload Payload for creating a run.
   * @returns The created run.
   */
  async create(
    threadId: string,
    assistantId: string,
    payload?: RunsCreatePayload,
  ): Promise<Run> {
    const json: Record<string, any> = {
      input: payload?.input,
      config: payload?.config,
      metadata: payload?.metadata,
      assistant_id: assistantId,
      interrupt_before: payload?.interruptBefore,
      interrupt_after: payload?.interruptAfter,
      webhook: payload?.webhook,
    };
    if (payload?.multitaskStrategy != null) {
      json["multitask_strategy"] = payload?.multitaskStrategy;
    }
    return this.fetch<Run>(`/threads/${threadId}/runs`, {
      method: "POST",
      json,
      signal: payload?.signal,
    });
  }

  async wait(
    threadId: null,
    assistantId: string,
    payload?: Omit<RunsWaitPayload, "multitaskStrategy">,
  ): Promise<ThreadState["values"]>;

  async wait(
    threadId: string,
    assistantId: string,
    payload?: RunsWaitPayload,
  ): Promise<ThreadState["values"]>;

  /**
   * Create a run and wait for it to complete.
   *
   * @param threadId The ID of the thread.
   * @param assistantId Assistant ID to use for this run.
   * @param payload Payload for creating a run.
   * @returns The last values chunk of the thread.
   */
  async wait(
    threadId: string | null,
    assistantId: string,
    payload?: RunsWaitPayload,
  ): Promise<ThreadState["values"]> {
    const json: Record<string, any> = {
      input: payload?.input,
      config: payload?.config,
      metadata: payload?.metadata,
      assistant_id: assistantId,
      interrupt_before: payload?.interruptBefore,
      interrupt_after: payload?.interruptAfter,
    };
    if (payload?.multitaskStrategy != null) {
      json["multitask_strategy"] = payload?.multitaskStrategy;
    }
    const endpoint =
      threadId == null ? `/runs/wait` : `/threads/${threadId}/runs/wait`;
    return this.fetch<ThreadState["values"]>(endpoint, {
      method: "POST",
      json,
      signal: payload?.signal,
    });
  }

  /**
   * List all runs for a thread.
   *
   * @param threadId The ID of the thread.
   * @param options Filtering and pagination options.
   * @returns List of runs.
   */
  async list(
    threadId: string,
    options?: {
      /**
       * Maximum number of runs to return.
       * Defaults to 10
       */
      limit?: number;

      /**
       * Offset to start from.
       * Defaults to 0.
       */
      offset?: number;
    },
  ): Promise<Run[]> {
    return this.fetch<Run[]>(`/threads/${threadId}/runs`, {
      params: {
        limit: options?.limit ?? 10,
        offset: options?.offset ?? 0,
      },
    });
  }

  /**
   * Get a run by ID.
   *
   * @param threadId The ID of the thread.
   * @param runId The ID of the run.
   * @returns The run.
   */
  async get(threadId: string, runId: string): Promise<Run> {
    return this.fetch<Run>(`/threads/${threadId}/runs/${runId}`);
  }

  /**
   * Cancel a run.
   *
   * @param threadId The ID of the thread.
   * @param runId The ID of the run.
   * @param wait Whether to block when canceling
   * @returns
   */
  async cancel(
    threadId: string,
    runId: string,
    wait: boolean = false,
  ): Promise<void> {
    return this.fetch<void>(`/threads/${threadId}/runs/${runId}/cancel`, {
      method: "POST",
      params: {
        wait: wait ? "1" : "0",
      },
    });
  }

  /**
   * Block until a run is done.
   *
   * @param threadId The ID of the thread.
   * @param runId The ID of the run.
   * @returns
   */
  async join(threadId: string, runId: string): Promise<void> {
    return this.fetch<void>(`/threads/${threadId}/runs/${runId}/join`);
  }

  /**
   * Delete a run.
   *
   * @param threadId The ID of the thread.
   * @param runId The ID of the run.
   * @returns
   */
  async delete(threadId: string, runId: string): Promise<void> {
    return this.fetch<void>(`/threads/${threadId}/runs/${runId}`, {
      method: "DELETE",
    });
  }
}

export class Client {
  /**
   * The client for interacting with assistants.
   */
  public assistants: AssistantsClient;

  /**
   * The client for interacting with threads.
   */
  public threads: ThreadsClient;

  /**
   * The client for interacting with runs.
   */
  public runs: RunsClient;

  /**
   * The client for interacting with cron runs.
   */
  public crons: CronsClient;

  constructor(config?: ClientConfig) {
    this.assistants = new AssistantsClient(config);
    this.threads = new ThreadsClient(config);
    this.runs = new RunsClient(config);
    this.crons = new CronsClient(config);
  }
}
