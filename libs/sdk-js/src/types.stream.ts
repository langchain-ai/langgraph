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

export type ValuesPayload<StateType> = { event: "values"; data: StateType };

/** @internal */
export type ValuesPayloadSubgraphs<StateType> = AsSubgraph<
  ValuesPayload<StateType>
>;

export type MessagesTuplePayload = {
  event: "messages";
  // TODO: add types for message and config, which do not depend on LangChain
  // while making sure it's easy to keep them in sync.
  data: [message: Message, config: MessageTupleMetadata];
};

/** @internal */
export type MessagesTuplePayloadSubgraphs = AsSubgraph<MessagesTuplePayload>;

export type MetadataPayload = {
  event: "metadata";
  data: { run_id: string; thread_id: string };
};

/** @internal */
export type MetadataPayloadSubgraphs = AsSubgraph<MetadataPayload>;

export type UpdatesPayload<UpdateType> = {
  event: "updates";
  data: { [node: string]: UpdateType };
};

/** @internal */
export type UpdatesPayloadSubgraphs<UpdateType> = AsSubgraph<
  UpdatesPayload<UpdateType>
>;

export type CustomPayload<T> = { event: "custom"; data: T };

/** @internal */
export type CustomPayloadSubgraphs<T> = AsSubgraph<CustomPayload<T>>;

type MessagesMetadataPayload = {
  event: "messages/metadata";
  data: { [messageId: string]: { metadata: unknown } };
};
type MessagesCompletePayload = {
  event: "messages/complete";
  data: Message[];
};
type MessagesPartialPayload = {
  event: "messages/partial";
  data: Message[];
};

export type MessagesPayload =
  | MessagesMetadataPayload
  | MessagesCompletePayload
  | MessagesPartialPayload;

/** @internal */
export type MessagesPayloadSubgraphs =
  | AsSubgraph<MessagesMetadataPayload>
  | AsSubgraph<MessagesCompletePayload>
  | AsSubgraph<MessagesPartialPayload>;

export type DebugPayload = { event: "debug"; data: unknown };

/** @internal */
export type DebugPayloadSubgraphs = AsSubgraph<DebugPayload>;

export type EventsPayload = { event: "events"; data: unknown };

/** @internal */
export type EventsPayloadSubgraphs = AsSubgraph<EventsPayload>;

type GetStreamModeMap<
  TStreamMode extends StreamMode | StreamMode[],
  TStateType = unknown,
  TUpdateType = TStateType,
  TCustomType = unknown,
> =
  | {
      values: ValuesPayload<TStateType>;
      updates: UpdatesPayload<TUpdateType>;
      custom: CustomPayload<TCustomType>;
      debug: DebugPayload;
      messages: MessagesPayload;
      "messages-tuple": MessagesTuplePayload;
      events: EventsPayload;
    }[TStreamMode extends StreamMode[] ? TStreamMode[number] : TStreamMode]
  | MetadataPayload;

type GetSubgraphsStreamModeMap<
  TStreamMode extends StreamMode | StreamMode[],
  TStateType = unknown,
  TUpdateType = TStateType,
  TCustomType = unknown,
> =
  | {
      values: ValuesPayloadSubgraphs<TStateType>;
      updates: UpdatesPayloadSubgraphs<TUpdateType>;
      custom: CustomPayloadSubgraphs<TCustomType>;
      debug: DebugPayloadSubgraphs;
      messages: MessagesPayloadSubgraphs;
      "messages-tuple": MessagesTuplePayloadSubgraphs;
      events: EventsPayloadSubgraphs;
    }[TStreamMode extends StreamMode[] ? TStreamMode[number] : TStreamMode]
  | MetadataPayloadSubgraphs;

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
