/**
 * EventSource/Server-Sent Events parser
 * @see https://html.spec.whatwg.org/multipage/server-sent-events.html
 *
 * Based on code from the {@link https://github.com/EventSource/eventsource | EventSource module},
 * which is licensed under the MIT license. And copyrighted the EventSource GitHub organisation.
 */
import type { EventSourceParseCallback, EventSourceParser } from "./types.js";

/**
 * Creates a new EventSource parser.
 *
 * @param onParse - Callback to invoke when a new event is parsed, or a new reconnection interval
 *                  has been sent from the server
 *
 * @returns A new EventSource parser, with `parse` and `reset` methods.
 * @public
 */
export function createParser(
  onParse: EventSourceParseCallback,
): EventSourceParser {
  // Processing state
  let isFirstChunk: boolean;
  let buffer: string;
  let startingPosition: number;
  let startingFieldLength: number;

  // Event state
  let eventId: string | undefined;
  let eventName: string | undefined;
  let data: string;

  reset();
  return { feed, reset };

  function reset(): void {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;

    eventId = undefined;
    eventName = undefined;
    data = "";
  }

  function feed(chunk: string): void {
    buffer = buffer ? buffer + chunk : chunk;

    // Strip any UTF8 byte order mark (BOM) at the start of the stream.
    // Note that we do not strip any non - UTF8 BOM, as eventsource streams are
    // always decoded as UTF8 as per the specification.
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }

    isFirstChunk = false;

    // Set up chunk-specific processing state
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;

    // Read the current buffer byte by byte
    while (position < length) {
      // EventSource allows for carriage return + line feed, which means we
      // need to ignore a linefeed character if the previous character was a
      // carriage return
      // @todo refactor to reduce nesting, consider checking previous byte?
      // @todo but consider multiple chunks etc
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }

      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character: string;

      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }

      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }

      parseEventStreamLine(buffer, position, fieldLength, lineLength);

      position += lineLength + 1;
    }

    if (position === length) {
      // If we consumed the entire buffer to read the event, reset the buffer
      buffer = "";
    } else if (position > 0) {
      // If there are bytes left to process, set the buffer to the unprocessed
      // portion of the buffer only
      buffer = buffer.slice(position);
    }
  }

  function parseEventStreamLine(
    lineBuffer: string,
    index: number,
    fieldLength: number,
    lineLength: number,
  ) {
    if (lineLength === 0) {
      // We reached the last line of this event
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || undefined,
          data: data.slice(0, -1), // remove trailing newline
        });

        data = "";
        eventId = undefined;
      }
      eventName = undefined;
      return;
    }

    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength),
    );
    let step = 0;

    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }

    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();

    if (field === "data") {
      data += value ? `${value}\n` : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\u0000")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({ type: "reconnect-interval", value: retry });
      }
    }
  }
}

const BOM = [239, 187, 191];

function hasBom(buffer: string) {
  return BOM.every(
    (charCode: number, index: number) => buffer.charCodeAt(index) === charCode,
  );
}
