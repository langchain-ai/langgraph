/* __LC_ALLOW_ENTRYPOINT_SIDE_EFFECTS__ */
"use client";

import { Client, type ClientConfig } from "../client.js";
import type {
  Command,
  DisconnectMode,
  MultitaskStrategy,
  OnCompletionBehavior,
} from "../types.js";
import type { Message } from "../types.messages.js";
import type {
  Checkpoint,
  Config,
  Interrupt,
  Metadata,
  ThreadState,
} from "../schema.js";
import type {
  CustomStreamEvent,
  DebugStreamEvent,
  ErrorStreamEvent,
  EventsStreamEvent,
  FeedbackStreamEvent,
  MessagesStreamEvent,
  MessagesTupleStreamEvent,
  MetadataStreamEvent,
  StreamMode,
  UpdatesStreamEvent,
  ValuesStreamEvent,
} from "../types.stream.js";

import {
  type MutableRefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type BaseMessageChunk,
  type BaseMessage,
  coerceMessageLikeToMessage,
  convertToChunk,
  isBaseMessageChunk,
} from "@langchain/core/messages";

class StreamError extends Error {
  constructor(data: { error?: string; name?: string; message: string }) {
    super(data.message);
    this.name = data.name ?? data.error ?? "StreamError";
  }

  static isStructuredError(error: unknown): error is {
    error?: string;
    name?: string;
    message: string;
  } {
    return typeof error === "object" && error != null && "message" in error;
  }
}

function tryConvertToChunk(message: BaseMessage): BaseMessageChunk | null {
  try {
    return convertToChunk(message);
  } catch {
    return null;
  }
}

class MessageTupleManager {
  chunks: Record<
    string,
    { chunk?: BaseMessageChunk | BaseMessage; index?: number }
  > = {};

  constructor() {
    this.chunks = {};
  }

  add(serialized: Message): string | null {
    // TODO: this is sometimes sent from the API
    // figure out how to prevent this or move this to LC.js
    if (serialized.type.endsWith("MessageChunk")) {
      serialized.type = serialized.type
        .slice(0, -"MessageChunk".length)
        .toLowerCase() as Message["type"];
    }

    const message = coerceMessageLikeToMessage(serialized);
    const chunk = tryConvertToChunk(message);

    const id = (chunk ?? message).id;
    if (!id) {
      console.warn(
        "No message ID found for chunk, ignoring in state",
        serialized,
      );
      return null;
    }

    this.chunks[id] ??= {};
    if (chunk) {
      const prev = this.chunks[id].chunk;
      this.chunks[id].chunk =
        (isBaseMessageChunk(prev) ? prev : null)?.concat(chunk) ?? chunk;
    } else {
      this.chunks[id].chunk = message;
    }

    return id;
  }

  clear() {
    this.chunks = {};
  }

  get(id: string, defaultIndex: number) {
    if (this.chunks[id] == null) return null;
    this.chunks[id].index ??= defaultIndex;

    return this.chunks[id];
  }
}

const toMessageDict = (chunk: BaseMessage): Message => {
  const { type, data } = chunk.toDict();
  return { ...data, type } as Message;
};

function unique<T>(array: T[]) {
  return [...new Set(array)] as T[];
}

function findLastIndex<T>(array: T[], predicate: (item: T) => boolean) {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i])) return i;
  }
  return -1;
}

interface Node<StateType = any> {
  type: "node";
  value: ThreadState<StateType>;
  path: string[];
}

interface Fork<StateType = any> {
  type: "fork";
  items: Array<Sequence<StateType>>;
}

interface Sequence<StateType = any> {
  type: "sequence";
  items: Array<Node<StateType> | Fork<StateType>>;
}

interface ValidFork<StateType = any> {
  type: "fork";
  items: Array<ValidSequence<StateType>>;
}

interface ValidSequence<StateType = any> {
  type: "sequence";
  items: [Node<StateType>, ...(Node<StateType> | ValidFork<StateType>)[]];
}

export type MessageMetadata<StateType extends Record<string, unknown>> = {
  /**
   * The ID of the message used.
   */
  messageId: string;

  /**
   * The first thread state the message was seen in.
   */
  firstSeenState: ThreadState<StateType> | undefined;

  /**
   * The branch of the message.
   */
  branch: string | undefined;

  /**
   * The list of branches this message is part of.
   * This is useful for displaying branching controls.
   */
  branchOptions: string[] | undefined;
};

function getBranchSequence<StateType extends Record<string, unknown>>(
  history: ThreadState<StateType>[],
) {
  const childrenMap: Record<string, ThreadState<StateType>[]> = {};

  // First pass - collect nodes for each checkpoint
  history.forEach((state) => {
    const checkpointId = state.parent_checkpoint?.checkpoint_id ?? "$";
    childrenMap[checkpointId] ??= [];
    childrenMap[checkpointId].push(state);
  });

  // Second pass - create a tree of sequences
  type Task = { id: string; sequence: Sequence; path: string[] };
  const rootSequence: Sequence = { type: "sequence", items: [] };
  const queue: Task[] = [{ id: "$", sequence: rootSequence, path: [] }];

  const paths: string[][] = [];

  const visited = new Set<string>();
  while (queue.length > 0) {
    const task = queue.shift()!;
    if (visited.has(task.id)) continue;
    visited.add(task.id);

    const children = childrenMap[task.id];
    if (children == null || children.length === 0) continue;

    // If we've encountered a fork (2+ children), push the fork
    // to the sequence and add a new sequence for each child
    let fork: Fork | undefined;
    if (children.length > 1) {
      fork = { type: "fork", items: [] };
      task.sequence.items.push(fork);
    }

    for (const value of children) {
      const id = value.checkpoint.checkpoint_id!;

      let sequence = task.sequence;
      let path = task.path;
      if (fork != null) {
        sequence = { type: "sequence", items: [] };
        fork.items.unshift(sequence);

        path = path.slice();
        path.push(id);
        paths.push(path);
      }

      sequence.items.push({ type: "node", value, path });
      queue.push({ id, sequence, path });
    }
  }

  return { rootSequence, paths };
}

const PATH_SEP = ">";
const ROOT_ID = "$";

// Get flat view
function getBranchView<StateType extends Record<string, unknown>>(
  sequence: Sequence<StateType>,
  paths: string[][],
  branch: string,
) {
  const path = branch.split(PATH_SEP);
  const pathMap: Record<string, string[][]> = {};

  for (const path of paths) {
    const parent = path.at(-2) ?? ROOT_ID;
    pathMap[parent] ??= [];
    pathMap[parent].unshift(path);
  }

  const history: ThreadState<StateType>[] = [];
  const branchByCheckpoint: Record<
    string,
    { branch: string | undefined; branchOptions: string[] | undefined }
  > = {};

  const forkStack = path.slice();
  const queue: (Node<StateType> | Fork<StateType>)[] = [...sequence.items];

  while (queue.length > 0) {
    const item = queue.shift()!;

    if (item.type === "node") {
      history.push(item.value);
      branchByCheckpoint[item.value.checkpoint.checkpoint_id!] = {
        branch: item.path.join(PATH_SEP),
        branchOptions: (item.path.length > 0
          ? pathMap[item.path.at(-2) ?? ROOT_ID] ?? []
          : []
        ).map((p) => p.join(PATH_SEP)),
      };
    }
    if (item.type === "fork") {
      const forkId = forkStack.shift();
      const index =
        forkId != null
          ? item.items.findIndex((value) => {
              const firstItem = value.items.at(0);
              if (!firstItem || firstItem.type !== "node") return false;
              return firstItem.value.checkpoint.checkpoint_id === forkId;
            })
          : -1;

      const nextItems = item.items.at(index)?.items ?? [];
      queue.push(...nextItems);
    }
  }

  return { history, branchByCheckpoint };
}

function fetchHistory<StateType extends Record<string, unknown>>(
  client: Client,
  threadId: string,
) {
  return client.threads.getHistory<StateType>(threadId, { limit: 1000 });
}

function useThreadHistory<StateType extends Record<string, unknown>>(
  threadId: string | undefined | null,
  client: Client,
  clearCallbackRef: MutableRefObject<(() => void) | undefined>,
  submittingRef: MutableRefObject<boolean>,
) {
  const [history, setHistory] = useState<ThreadState<StateType>[]>([]);

  const fetcher = useCallback(
    (
      threadId: string | undefined | null,
    ): Promise<ThreadState<StateType>[]> => {
      if (threadId != null) {
        return fetchHistory<StateType>(client, threadId).then((history) => {
          setHistory(history);
          return history;
        });
      }

      setHistory([]);
      clearCallbackRef.current?.();
      return Promise.resolve([]);
    },
    [],
  );

  useEffect(() => {
    if (submittingRef.current) return;
    fetcher(threadId);
  }, [fetcher, submittingRef, threadId]);

  return {
    data: history,
    mutate: (mutateId?: string) => fetcher(mutateId ?? threadId),
  };
}

const useControllableThreadId = (options?: {
  threadId?: string | null;
  onThreadId?: (threadId: string) => void;
}): [string | null, (threadId: string) => void] => {
  const [localThreadId, _setLocalThreadId] = useState<string | null>(
    options?.threadId ?? null,
  );

  const onThreadIdRef = useRef(options?.onThreadId);
  onThreadIdRef.current = options?.onThreadId;

  const onThreadId = useCallback((threadId: string) => {
    _setLocalThreadId(threadId);
    onThreadIdRef.current?.(threadId);
  }, []);

  if (!options || !("threadId" in options)) {
    return [localThreadId, onThreadId];
  }

  return [options.threadId ?? null, onThreadId];
};

type BagTemplate = {
  ConfigurableType?: Record<string, unknown>;
  InterruptType?: unknown;
  CustomEventType?: unknown;
  UpdateType?: unknown;
};

type GetUpdateType<
  Bag extends BagTemplate,
  StateType extends Record<string, unknown>,
> = Bag extends { UpdateType: unknown }
  ? Bag["UpdateType"]
  : Partial<StateType>;

type GetConfigurableType<Bag extends BagTemplate> = Bag extends {
  ConfigurableType: Record<string, unknown>;
}
  ? Bag["ConfigurableType"]
  : Record<string, unknown>;

type GetInterruptType<Bag extends BagTemplate> = Bag extends {
  InterruptType: unknown;
}
  ? Bag["InterruptType"]
  : unknown;

type GetCustomEventType<Bag extends BagTemplate> = Bag extends {
  CustomEventType: unknown;
}
  ? Bag["CustomEventType"]
  : unknown;

interface UseStreamOptions<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  Bag extends BagTemplate = BagTemplate,
> {
  /**
   * The ID of the assistant to use.
   */
  assistantId: string;

  /**
   * The URL of the API to use.
   */
  apiUrl: ClientConfig["apiUrl"];

  /**
   * The API key to use.
   */
  apiKey?: ClientConfig["apiKey"];

  /**
   * Specify the key within the state that contains messages.
   * Defaults to "messages".
   *
   * @default "messages"
   */
  messagesKey?: string;

  /**
   * Callback that is called when an error occurs.
   */
  onError?: (error: unknown) => void;

  /**
   * Callback that is called when the stream is finished.
   */
  onFinish?: (state: ThreadState<StateType>) => void;

  /**
   * Callback that is called when an update event is received.
   */
  onUpdateEvent?: (
    data: UpdatesStreamEvent<GetUpdateType<Bag, StateType>>["data"],
  ) => void;

  /**
   * Callback that is called when a custom event is received.
   */
  onCustomEvent?: (
    data: CustomStreamEvent<GetCustomEventType<Bag>>["data"],
  ) => void;

  /**
   * Callback that is called when a metadata event is received.
   */
  onMetadataEvent?: (data: MetadataStreamEvent["data"]) => void;

  /**
   * The ID of the thread to fetch history and current values from.
   */
  threadId?: string | null;

  /**
   * Callback that is called when the thread ID is updated (ie when a new thread is created).
   */
  onThreadId?: (threadId: string) => void;
}

interface UseStream<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  Bag extends BagTemplate = BagTemplate,
> {
  /**
   * The current values of the thread.
   */
  values: StateType;

  /**
   * Last seen error from the thread or during streaming.
   */
  error: unknown;

  /**
   * Whether the stream is currently running.
   */
  isLoading: boolean;

  /**
   * Stops the stream.
   */
  stop: () => void;

  /**
   * Create and stream a run to the thread.
   */
  submit: (
    values: GetUpdateType<Bag, StateType> | null | undefined,
    options?: SubmitOptions<StateType, GetConfigurableType<Bag>>,
  ) => void;

  /**
   * The current branch of the thread.
   */
  branch: string;

  /**
   * Set the branch of the thread.
   */
  setBranch: (branch: string) => void;

  /**
   * Flattened history of thread states of a thread.
   */
  history: ThreadState<StateType>[];

  /**
   * Tree of all branches for the thread.
   * @experimental
   */
  experimental_branchTree: Sequence<StateType>;

  /**
   * Get the interrupt value for the stream if interrupted.
   */
  interrupt: Interrupt<GetInterruptType<Bag>> | undefined;

  /**
   * Messages inferred from the thread.
   * Will automatically update with incoming message chunks.
   */
  messages: Message[];

  /**
   * Get the metadata for a message, such as first thread state the message
   * was seen in and branch information.
   
   * @param message - The message to get the metadata for.
   * @param index - The index of the message in the thread.
   * @returns The metadata for the message.
   */
  getMessagesMetadata: (
    message: Message,
    index?: number,
  ) => MessageMetadata<StateType> | undefined;
}

type ConfigWithConfigurable<ConfigurableType extends Record<string, unknown>> =
  Config & { configurable?: ConfigurableType };

interface SubmitOptions<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  ConfigurableType extends Record<string, unknown> = Record<string, unknown>,
> {
  config?: ConfigWithConfigurable<ConfigurableType>;
  checkpoint?: Omit<Checkpoint, "thread_id"> | null;
  command?: Command;
  interruptBefore?: "*" | string[];
  interruptAfter?: "*" | string[];
  metadata?: Metadata;
  multitaskStrategy?: MultitaskStrategy;
  onCompletion?: OnCompletionBehavior;
  onDisconnect?: DisconnectMode;
  feedbackKeys?: string[];
  streamMode?: Array<StreamMode>;
  optimisticValues?:
    | Partial<StateType>
    | ((prev: StateType) => Partial<StateType>);
}

export function useStream<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  Bag extends {
    ConfigurableType?: Record<string, unknown>;
    InterruptType?: unknown;
    CustomEventType?: unknown;
    UpdateType?: unknown;
  } = BagTemplate,
>(options: UseStreamOptions<StateType, Bag>): UseStream<StateType, Bag> {
  type UpdateType = GetUpdateType<Bag, StateType>;
  type CustomType = GetCustomEventType<Bag>;
  type InterruptType = GetInterruptType<Bag>;
  type ConfigurableType = GetConfigurableType<Bag>;

  type EventStreamEvent =
    | ValuesStreamEvent<StateType>
    | UpdatesStreamEvent<UpdateType>
    | CustomStreamEvent<CustomType>
    | DebugStreamEvent
    | MessagesStreamEvent
    | MessagesTupleStreamEvent
    | EventsStreamEvent
    | MetadataStreamEvent
    | ErrorStreamEvent
    | FeedbackStreamEvent;

  let { assistantId, messagesKey, onError, onFinish } = options;
  messagesKey ??= "messages";

  const client = useMemo(
    () => new Client({ apiUrl: options.apiUrl, apiKey: options.apiKey }),
    [options.apiKey, options.apiUrl],
  );
  const [threadId, onThreadId] = useControllableThreadId(options);

  const [branch, setBranch] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const [streamError, setStreamError] = useState<unknown>(undefined);
  const [streamValues, setStreamValues] = useState<StateType | null>(null);

  const messageManagerRef = useRef(new MessageTupleManager());
  const submittingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const trackStreamModeRef = useRef<
    Array<"values" | "updates" | "events" | "custom" | "messages-tuple">
  >([]);

  const trackStreamMode = useCallback(
    (mode: Exclude<StreamMode, "debug" | "messages">) => {
      if (!trackStreamModeRef.current.includes(mode))
        trackStreamModeRef.current.push(mode);
    },
    [],
  );

  const hasUpdateListener = options.onUpdateEvent != null;
  const hasCustomListener = options.onCustomEvent != null;

  const callbackStreamMode = useMemo(() => {
    const modes: Exclude<StreamMode, "debug" | "messages">[] = [];
    if (hasUpdateListener) modes.push("updates");
    if (hasCustomListener) modes.push("custom");
    return modes;
  }, [hasUpdateListener, hasCustomListener]);

  const clearCallbackRef = useRef<() => void>(null!);
  clearCallbackRef.current = () => {
    setStreamError(undefined);
    setStreamValues(null);
  };

  // TODO: this should be done on the server to avoid pagination
  // TODO: should we permit adapter? SWR / React Query?
  const history = useThreadHistory<StateType>(
    threadId,
    client,
    clearCallbackRef,
    submittingRef,
  );

  const getMessages = useMemo(() => {
    return (value: StateType) =>
      Array.isArray(value[messagesKey])
        ? (value[messagesKey] as Message[])
        : [];
  }, [messagesKey]);

  const { rootSequence, paths } = getBranchSequence(history.data);
  const { history: flatHistory, branchByCheckpoint } = getBranchView(
    rootSequence,
    paths,
    branch,
  );

  const threadHead: ThreadState<StateType> | undefined = flatHistory.at(-1);
  const historyValues = threadHead?.values ?? ({} as StateType);
  const historyError = (() => {
    const error = threadHead?.tasks?.at(-1)?.error;
    if (error == null) return undefined;
    try {
      const parsed = JSON.parse(error) as unknown;
      if (StreamError.isStructuredError(parsed)) {
        return new StreamError(parsed);
      }

      return parsed;
    } catch {
      // do nothing
    }
    return error;
  })();

  const messageMetadata = (() => {
    const alreadyShown = new Set<string>();
    return getMessages(historyValues).map(
      (message, idx): MessageMetadata<StateType> => {
        const messageId = message.id ?? idx;
        const firstSeenIdx = findLastIndex(history.data, (state) =>
          getMessages(state.values)
            .map((m, idx) => m.id ?? idx)
            .includes(messageId),
        );

        const firstSeen = history.data[firstSeenIdx] as
          | ThreadState<StateType>
          | undefined;

        let branch = firstSeen
          ? branchByCheckpoint[firstSeen.checkpoint.checkpoint_id!]
          : undefined;

        if (!branch?.branch?.length) branch = undefined;

        // serialize branches
        const optionsShown = branch?.branchOptions?.flat(2).join(",");
        if (optionsShown) {
          if (alreadyShown.has(optionsShown)) branch = undefined;
          alreadyShown.add(optionsShown);
        }

        return {
          messageId: messageId.toString(),
          firstSeenState: firstSeen,

          branch: branch?.branch,
          branchOptions: branch?.branchOptions,
        };
      },
    );
  })();

  const stop = useCallback(() => {
    if (abortRef.current != null) abortRef.current.abort();
    abortRef.current = null;
  }, []);

  const submit = async (
    values: UpdateType | null | undefined,
    submitOptions?: SubmitOptions<StateType, ConfigurableType>,
  ) => {
    try {
      setIsLoading(true);
      setStreamError(undefined);

      submittingRef.current = true;
      abortRef.current = new AbortController();

      let usableThreadId = threadId;
      if (!usableThreadId) {
        const thread = await client.threads.create();
        onThreadId(thread.thread_id);
        usableThreadId = thread.thread_id;
      }

      const streamMode = unique([
        ...(submitOptions?.streamMode ?? []),
        ...trackStreamModeRef.current,
        ...callbackStreamMode,
      ]);

      const checkpoint =
        submitOptions?.checkpoint ?? threadHead?.checkpoint ?? undefined;
      // @ts-expect-error
      if (checkpoint != null) delete checkpoint.thread_id;

      const run = (await client.runs.stream(usableThreadId, assistantId, {
        input: values as Record<string, unknown>,
        config: submitOptions?.config,
        command: submitOptions?.command,

        interruptBefore: submitOptions?.interruptBefore,
        interruptAfter: submitOptions?.interruptAfter,
        metadata: submitOptions?.metadata,
        multitaskStrategy: submitOptions?.multitaskStrategy,
        onCompletion: submitOptions?.onCompletion,
        onDisconnect: submitOptions?.onDisconnect ?? "cancel",

        signal: abortRef.current.signal,

        checkpoint,
        streamMode,
      })) as AsyncGenerator<EventStreamEvent>;

      // Unbranch things
      const newPath = submitOptions?.checkpoint?.checkpoint_id
        ? branchByCheckpoint[submitOptions?.checkpoint?.checkpoint_id]?.branch
        : undefined;

      if (newPath != null) setBranch(newPath ?? "");

      // Assumption: we're setting the initial value
      // Used for instant feedback
      setStreamValues(() => {
        const values = { ...historyValues };

        if (submitOptions?.optimisticValues != null) {
          return {
            ...values,
            ...(typeof submitOptions.optimisticValues === "function"
              ? submitOptions.optimisticValues(values)
              : submitOptions.optimisticValues),
          };
        }

        return values;
      });

      let streamError: StreamError | undefined;
      for await (const { event, data } of run) {
        if (event === "error") {
          streamError = new StreamError(data);
          break;
        }

        if (event === "updates") options.onUpdateEvent?.(data);
        if (event === "custom") options.onCustomEvent?.(data);
        if (event === "metadata") options.onMetadataEvent?.(data);

        if (event === "values") setStreamValues(data);
        if (event === "messages") {
          const [serialized] = data;

          const messageId = messageManagerRef.current.add(serialized);
          if (!messageId) {
            console.warn(
              "Failed to add message to manager, no message ID found",
            );
            continue;
          }

          setStreamValues((streamValues) => {
            const values = { ...historyValues, ...streamValues };

            // Assumption: we're concatenating the message
            const messages = getMessages(values).slice();
            const { chunk, index } =
              messageManagerRef.current.get(messageId, messages.length) ?? {};

            if (!chunk || index == null) return values;
            messages[index] = toMessageDict(chunk);

            return { ...values, [messagesKey!]: messages };
          });
        }
      }

      // TODO: stream created checkpoints to avoid an unnecessary network request
      const result = await history.mutate(usableThreadId);
      setStreamValues(null);

      if (streamError != null) throw streamError;

      const lastHead = result.at(0);
      if (lastHead) onFinish?.(lastHead);
    } catch (error) {
      if (
        !(
          error instanceof Error &&
          (error.name === "AbortError" || error.name === "TimeoutError")
        )
      ) {
        console.error(error);
        setStreamError(error);
        onError?.(error);
      }
    } finally {
      setIsLoading(false);

      // Assumption: messages are already handled, we can clear the manager
      messageManagerRef.current.clear();
      submittingRef.current = false;
      abortRef.current = null;
    }
  };

  const error = streamError ?? historyError;
  const values = streamValues ?? historyValues;

  return {
    get values() {
      trackStreamMode("values");
      return values;
    },

    error,
    isLoading,

    stop,
    submit,

    branch,
    setBranch,

    history: flatHistory,
    experimental_branchTree: rootSequence,

    get interrupt() {
      // Don't show the interrupt if the stream is loading
      if (isLoading) return undefined;

      const interrupts = threadHead?.tasks?.at(-1)?.interrupts;
      if (interrupts == null || interrupts.length === 0) {
        // check if there's a next task present
        const next = threadHead?.next ?? [];
        if (!next.length || error != null) return undefined;
        return { when: "breakpoint" };
      }

      // Return only the current interrupt
      return interrupts.at(-1) as Interrupt<InterruptType> | undefined;
    },

    get messages() {
      trackStreamMode("messages-tuple");
      return getMessages(values);
    },

    getMessagesMetadata(
      message: Message,
      index?: number,
    ): MessageMetadata<StateType> | undefined {
      trackStreamMode("messages-tuple");
      return messageMetadata?.find(
        (m) => m.messageId === (message.id ?? index),
      );
    },
  };
}
