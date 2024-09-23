/**
 * EventSource parser instance.
 *
 * Needs to be reset between reconnections/when switching data source, using the `reset()` method.
 *
 * @public
 */
export interface EventSourceParser {
  /**
   * Feeds the parser another chunk. The method _does not_ return a parsed message.
   * Instead, if the chunk was a complete message (or completed a previously incomplete message),
   * it will invoke the `onParse` callback used to create the parsers.
   *
   * @param chunk - The chunk to parse. Can be a partial, eg in the case of streaming messages.
   * @public
   */
  feed(chunk: string): void;

  /**
   * Resets the parser state. This is required when you have a new stream of messages -
   * for instance in the case of a client being disconnected and reconnecting.
   *
   * @public
   */
  reset(): void;
}

/**
 * A parsed EventSource event
 *
 * @public
 */
export interface ParsedEvent {
  /**
   * Differentiates the type from reconnection intervals and other types of messages
   * Not to be confused with `event`.
   */
  type: "event";

  /**
   * The event type sent from the server. Note that this differs from the browser `EventSource`
   * implementation in that browsers will default this to `message`, whereas this parser will
   * leave this as `undefined` if not explicitly declared.
   */
  event?: string;

  /**
   * ID of the message, if any was provided by the server. Can be used by clients to keep the
   * last received message ID in sync when reconnecting.
   */
  id?: string;

  /**
   * The data received for this message
   */
  data: string;
}

/**
 * An event emitted from the parser when the server sends a value in the `retry` field,
 * indicating how many seconds the client should wait before attempting to reconnect.
 *
 * @public
 */
export interface ReconnectInterval {
  /**
   * Differentiates the type from `event` and other types of messages
   */
  type: "reconnect-interval";

  /**
   * Number of seconds to wait before reconnecting. Note that the parser does not care about
   * this value at all - it only emits the value for clients to use.
   */
  value: number;
}

/**
 * The different types of messages the parsed can emit to the `onParse` callback
 *
 * @public
 */
export type ParseEvent = ParsedEvent | ReconnectInterval;

/**
 * Callback passed as the `onParse` callback to a parser
 *
 * @public
 */
export type EventSourceParseCallback = (event: ParseEvent) => void;
