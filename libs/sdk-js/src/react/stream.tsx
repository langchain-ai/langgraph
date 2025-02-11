/* __LC_ALLOW_ENTRYPOINT_SIDE_EFFECTS__ */
"use client";

import { Client } from "../client.js";
import type { Command } from "../types.js";
import type { Message } from "../types.messages.js";
import type { Checkpoint, Config, ThreadState } from "../schema.js";
import type {
  CustomStreamEvent,
  DebugStreamEvent,
  ErrorStreamEvent,
  EventsStreamEvent,
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
} from "@langchain/core/messages";

class StreamError extends Error {
  constructor(data: { error: string; message: string }) {
    super([data.error, data.message].filter(Boolean).join(": "));
  }
}

class MessageTupleManager {
  chunks: Record<string, { chunk?: BaseMessageChunk; index?: number }> = {};

  constructor() {
    this.chunks = {};
  }

  add(serialized: Message): string | null {
    const chunk = convertToChunk(coerceMessageLikeToMessage(serialized));

    const id = chunk.id;
    if (!id) return null;

    this.chunks[id] ??= {};
    this.chunks[id].chunk = this.chunks[id]?.chunk?.concat(chunk) ?? chunk;

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
  messageId: string;
  firstSeenState: ThreadState<StateType> | undefined;

  branch: string | undefined;
  branchOptions: string[] | undefined;
};

function fetchHistory<StateType extends Record<string, unknown>>(
  client: Client,
  threadId: string,
) {
  return client.threads.getHistory<StateType>(threadId, { limit: 1000 });
}

function useThreadHistory<StateType extends Record<string, unknown>>(
  threadId: string | undefined | null,
  client: Client,
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

export function useStream<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  UpdateType extends Record<string, unknown> = Partial<StateType>,
  CustomType = unknown,
>(options: {
  assistantId: string;
  client: Client<StateType, UpdateType, CustomType>;

  withMessages?: string;

  onError?: (error: unknown) => void;
  onFinish?: (state: ThreadState<StateType>) => void;

  // TODO: can we make threadId uncontrollable / controllable?
  threadId?: string | null;
  onThreadId?: (threadId: string) => void;
}) {
  type EventStreamEvent =
    | ValuesStreamEvent<StateType>
    | UpdatesStreamEvent<UpdateType>
    | CustomStreamEvent<CustomType>
    | DebugStreamEvent
    | MessagesStreamEvent
    | MessagesTupleStreamEvent
    | EventsStreamEvent
    | MetadataStreamEvent
    | ErrorStreamEvent;

  const { assistantId, threadId, client, withMessages, onError, onFinish } =
    options;

  if (client == null) {
    throw new Error(
      "LangGraph SDK not provided. Either pass a client to `useStream` or wrap your app in a `LangGraphConfig` provider and pass the client there.",
    );
  }

  const [branchPath, setBranchPath] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<unknown | undefined>(undefined);
  const [events, setEvents] = useState<EventStreamEvent[]>([]);

  const [streamValues, setStreamValues] = useState<StateType | null>(null);

  const messageManagerRef = useRef(new MessageTupleManager());
  const submittingRef = useRef(false);
  const trackStreamModeRef = useRef<
    Array<"values" | "updates" | "events" | "custom" | "messages-tuple">
  >(["values", "messages-tuple"]);

  const trackStreamMode = useCallback(
    (mode: Exclude<StreamMode, "debug" | "messages">) => {
      if (!trackStreamModeRef.current.includes(mode))
        trackStreamModeRef.current.push(mode);
    },
    [],
  );

  // TODO: this should be done on the server to avoid pagination
  // TODO: should we permit adapter? SWR / React Query?
  const history = useThreadHistory<StateType>(threadId, client, submittingRef);

  const getMessages = useMemo(() => {
    if (withMessages == null) return undefined;
    return (value: StateType) =>
      Array.isArray(value[withMessages])
        ? (value[withMessages] as Message[])
        : [];
  }, [withMessages]);

  const [sequence, pathMap] = (() => {
    const childrenMap: Record<string, ThreadState<StateType>[]> = {};

    // First pass - collect nodes for each checkpoint
    history.data.forEach((state) => {
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

    // Third pass, create a map for available forks
    const pathMap: Record<string, string[][]> = {};
    for (const path of paths) {
      const parent = path.at(-2) ?? "$";
      pathMap[parent] ??= [];
      pathMap[parent].unshift(path);
    }

    return [rootSequence as ValidSequence, pathMap];
  })();

  const [flatValues, flatPaths] = (() => {
    const result: ThreadState<StateType>[] = [];

    // TODO: this is kinda ugly
    const flatPaths: Record<
      string,
      { current: string[] | undefined; branches: string[][] | undefined }
    > = {};

    const forkStack = branchPath.slice();
    const queue: (Node<StateType> | Fork<StateType>)[] = [...sequence.items];

    while (queue.length > 0) {
      const item = queue.shift()!;

      if (item.type === "node") {
        result.push(item.value);
        flatPaths[item.value.checkpoint.checkpoint_id!] = {
          current: item.path,
          branches:
            item.path.length > 0 ? pathMap[item.path.at(-2) ?? "$"] ?? [] : [],
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

    return [result, flatPaths];
  })();

  const threadHead: ThreadState<StateType> | undefined = flatValues.at(-1);
  const historyValues = threadHead?.values ?? ({} as StateType);

  const messageMetadata = (() => {
    if (getMessages == null) return undefined;

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
          ? flatPaths[firstSeen.checkpoint.checkpoint_id!]
          : undefined;

        if (!branch?.current?.length) branch = undefined;

        // serialize branches
        const optionsShown = branch?.branches?.flat(2).join(",");
        if (optionsShown) {
          if (alreadyShown.has(optionsShown)) branch = undefined;
          alreadyShown.add(optionsShown);
        }

        return {
          messageId: messageId.toString(),
          firstSeenState: firstSeen,
          branch: branch?.current?.join(">"),
          branchOptions: branch?.branches?.map((b) => b.join(">")),
        };
      },
    );
  })();

  const handleSubmit = async (
    values: UpdateType | undefined,
    submitOptions?: {
      config?: Config;
      checkpoint?: Omit<Checkpoint, "thread_id"> | null;
      command?: Command;
      streamMode?: Array<StreamMode>;
      optimisticValues?:
        | Partial<StateType>
        | ((prev: StateType) => Partial<StateType>);
    },
  ) => {
    try {
      setIsLoading(true);
      submittingRef.current = true;

      let usableThreadId = threadId;
      if (!usableThreadId) {
        const thread = await client.threads.create();
        options?.onThreadId?.(thread.thread_id);
        usableThreadId = thread.thread_id;
      }

      const streamMode = unique([
        ...(submitOptions?.streamMode ?? []),
        ...trackStreamModeRef.current,
      ]);

      const checkpoint =
        submitOptions?.checkpoint ?? threadHead?.checkpoint ?? undefined;
      // @ts-expect-error
      if (checkpoint != null) delete checkpoint.thread_id;

      // TODO: why non-existent assistant ID does not throw an error here?
      const run = (await client.runs.stream(usableThreadId, assistantId, {
        input: values as Record<string, unknown>,
        config: submitOptions?.config,
        checkpoint,
        streamMode,
      })) as AsyncGenerator<EventStreamEvent>;

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

      for await (const { event, data } of run) {
        setEvents((events) => [...events, { event, data } as EventStreamEvent]);

        if (event === "error") {
          const error = new StreamError(data);
          setError(error);
          onError?.(error);
          break;
        }

        if (event === "values") {
          setStreamValues(data);
        } else if (event === "messages") {
          if (!getMessages) continue;

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

            // Assumption: we're concating the message
            const messages = getMessages(values).slice();
            const { chunk, index } =
              messageManagerRef.current.get(messageId, messages.length) ?? {};

            if (!chunk || index == null) return values;
            messages[index] = toMessageDict(chunk);

            return { ...values, [withMessages!]: messages };
          });
        }
      }

      // TODO: stream created checkpoints to avoid an unnecessary network request
      const result = await history.mutate(usableThreadId);
      setStreamValues(null);

      const lastHead = result.at(0);
      if (lastHead) onFinish?.(lastHead);
    } catch (error) {
      setError(error);
      onError?.(error);
    } finally {
      setIsLoading(false);

      // Assumption: messages are already handled, we can clear the manager
      messageManagerRef.current.clear();
      submittingRef.current = false;
    }
  };

  const values = streamValues ?? historyValues;
  const stream = {
    get custom() {
      trackStreamMode("custom");

      return events
        .filter((item) => item.event === "custom")
        .map(({ data }) => data as CustomType);
    },

    get events() {
      trackStreamMode("events");
      return events;
    },

    get updates() {
      trackStreamMode("updates");
      return events
        .filter(
          (item): item is UpdatesStreamEvent<UpdateType> =>
            item.event === "updates",
        )
        .map(({ data }) => data);
    },
  };

  const setBranch = useCallback(
    (path: string) => setBranchPath(path.split(">")),
    [setBranchPath],
  );

  return {
    get values() {
      trackStreamMode("values");
      return values;
    },

    error,
    isLoading,

    handleSubmit,
    setBranch,

    stream,

    get messages() {
      trackStreamMode("messages-tuple");

      if (getMessages == null) {
        throw new Error(
          "No messages key provided. Make sure that `useStream` contains the `messagesKey` property.",
        );
      }

      return getMessages(values);
    },

    getMessagesMetadata(
      message: Message,
      index?: number,
    ): MessageMetadata<StateType> | undefined {
      trackStreamMode("messages-tuple");

      if (getMessages == null) {
        throw new Error(
          "No messages key provided. Make sure that `useStream` contains the `messagesKey` property.",
        );
      }

      return messageMetadata?.find(
        (m) => m.messageId === (message.id ?? index),
      );
    },
  };
}
