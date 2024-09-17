import { createParser } from "./parse.js";
import type { EventSourceParser, ParsedEvent } from "./types.js";

/**
 * A TransformStream that ingests a stream of strings and produces a stream of ParsedEvents.
 *
 * @example
 * ```
 * const eventStream =
 *   response.body
 *     .pipeThrough(new TextDecoderStream())
 *     .pipeThrough(new EventSourceParserStream())
 * ```
 * @public
 */
export class EventSourceParserStream extends TransformStream<
  string,
  ParsedEvent
> {
  constructor() {
    let parser!: EventSourceParser;

    super({
      start(controller) {
        parser = createParser((event: any) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },
      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

export type { ParsedEvent } from "./types.js";
