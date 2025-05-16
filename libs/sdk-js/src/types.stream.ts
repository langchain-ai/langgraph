import type { Message } from "./types.messages.js";

/**
 * Stream modes
 * - "values": Stream only the state values.
 * - "messages": Stream complete messages.
 * - "messages-tuple": Stream (message chunk, metadata) tuples.
 * - "updates": Stream updates to the state.
 * - "events": Stream events occurring during execution.
 * - "debug": Stream detailed debug information.
 * - "custom": Stream custom events.
 */
export type StreamMode =
  | "values"
  | "messages"
  | "updates"
  | "events"
  | "debug"
  | "custom"
  | "messages-tuple";

type MessageTupleMetadata = {
  tags: string[];
  [key: string]: unknown;
};

type AsSubgraph<TEvent extends { event: string; data: unknown }> = {
  event: TEvent["event"] | `${TEvent["event"]}|${string}`;
  data: TEvent["data"];
};

/**
 * Stream event with values after completion of each step.
 */
export type ValuesStreamEvent<StateType> = { event: "values"; data: StateType };

/** @internal */
export type SubgraphValuesStreamEvent<StateType> = AsSubgraph<
  ValuesStreamEvent<StateType>
>;

/**
 * Stream event with message chunks coming from LLM invocations inside nodes.
 */
export type MessagesTupleStreamEvent = {
  event: "messages";
  // TODO: add types for message and config, which do not depend on LangChain
  // while making sure it's easy to keep them in sync.
  data: [message: Message, config: MessageTupleMetadata];
};

/** @internal */
export type SubgraphMessagesTupleStreamEvent =
  AsSubgraph<MessagesTupleStreamEvent>;

/**
 * Metadata stream event with information about the run and thread
 */
export type MetadataStreamEvent = {
  event: "metadata";
  data: { run_id: string; thread_id: string };
};

/**
 * Stream event with error information.
 */
export type ErrorStreamEvent = {
  event: "error";
  data: { error: string; message: string };
};

/** @internal */
export type SubgraphErrorStreamEvent = AsSubgraph<ErrorStreamEvent>;

/**
 * Stream event with updates to the state after each step.
 * The streamed outputs include the name of the node that
 * produced the update as well as the update.
 */
export type UpdatesStreamEvent<UpdateType> = {
  event: "updates";
  data: { [node: string]: UpdateType };
};

/** @internal */
export type SubgraphUpdatesStreamEvent<UpdateType> = AsSubgraph<
  UpdatesStreamEvent<UpdateType>
>;

/**
 * Streaming custom data from inside the nodes.
 */
export type CustomStreamEvent<T> = { event: "custom"; data: T };

/** @internal */
export type SubgraphCustomStreamEvent<T> = AsSubgraph<CustomStreamEvent<T>>;

type MessagesMetadataStreamEvent = {
  event: "messages/metadata";
  data: { [messageId: string]: { metadata: unknown } };
};
type MessagesCompleteStreamEvent = {
  event: "messages/complete";
  data: Message[];
};
type MessagesPartialStreamEvent = {
  event: "messages/partial";
  data: Message[];
};

/**
 * Message stream event specific to LangGraph Server.
 * @deprecated Use `streamMode: "messages-tuple"` instead.
 */
export type MessagesStreamEvent =
  | MessagesMetadataStreamEvent
  | MessagesCompleteStreamEvent
  | MessagesPartialStreamEvent;

/** @internal */
export type SubgraphMessagesStreamEvent =
  | AsSubgraph<MessagesMetadataStreamEvent>
  | AsSubgraph<MessagesCompleteStreamEvent>
  | AsSubgraph<MessagesPartialStreamEvent>;

/**
 * Stream event with detailed debug information.
 */
export type DebugStreamEvent = { event: "debug"; data: unknown };

/** @internal */
export type SubgraphDebugStreamEvent = AsSubgraph<DebugStreamEvent>;

/**
 * Stream event with events occurring during execution.
 */
export type EventsStreamEvent = {
  event: "events";
  data: {
    event:
      | `on_${"chat_model" | "llm" | "chain" | "tool" | "retriever" | "prompt"}_${"start" | "stream" | "end"}`
      | (string & {});
    name: string;
    tags: string[];
    run_id: string;
    metadata: Record<string, unknown>;
    parent_ids: string[];
    data: unknown;
  };
};

/** @internal */
export type SubgraphEventsStreamEvent = AsSubgraph<EventsStreamEvent>;

/**
 * Stream event with a feedback key to signed URL map. Set `feedbackKeys` in
 * the `RunsStreamPayload` to receive this event.
 */
export type FeedbackStreamEvent = {
  event: "feedback";
  data: { [feedbackKey: string]: string };
};

type GetStreamModeMap<
  TStreamMode extends StreamMode | StreamMode[],
  TStateType = unknown,
  TUpdateType = TStateType,
  TCustomType = unknown,
> =
  | {
      values: ValuesStreamEvent<TStateType>;
      updates: UpdatesStreamEvent<TUpdateType>;
      custom: CustomStreamEvent<TCustomType>;
      debug: DebugStreamEvent;
      messages: MessagesStreamEvent;
      "messages-tuple": MessagesTupleStreamEvent;
      events: EventsStreamEvent;
    }[TStreamMode extends StreamMode[] ? TStreamMode[number] : TStreamMode]
  | ErrorStreamEvent
  | MetadataStreamEvent
  | FeedbackStreamEvent;

type GetSubgraphsStreamModeMap<
  TStreamMode extends StreamMode | StreamMode[],
  TStateType = unknown,
  TUpdateType = TStateType,
  TCustomType = unknown,
> =
  | {
      values: SubgraphValuesStreamEvent<TStateType>;
      updates: SubgraphUpdatesStreamEvent<TUpdateType>;
      custom: SubgraphCustomStreamEvent<TCustomType>;
      debug: SubgraphDebugStreamEvent;
      messages: SubgraphMessagesStreamEvent;
      "messages-tuple": SubgraphMessagesTupleStreamEvent;
      events: SubgraphEventsStreamEvent;
    }[TStreamMode extends StreamMode[] ? TStreamMode[number] : TStreamMode]
  | SubgraphErrorStreamEvent
  | MetadataStreamEvent
  | FeedbackStreamEvent;

export type TypedAsyncGenerator<
  TStreamMode extends StreamMode | StreamMode[] = [],
  TSubgraphs extends boolean = false,
  TStateType = unknown,
  TUpdateType = TStateType,
  TCustomType = unknown,
> = AsyncGenerator<
  TSubgraphs extends true
    ? GetSubgraphsStreamModeMap<
        TStreamMode,
        TStateType,
        TUpdateType,
        TCustomType
      >
    : GetStreamModeMap<TStreamMode, TStateType, TUpdateType, TCustomType>
>;
