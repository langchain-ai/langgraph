// From https://github.com/rexxars/eventsource-parser
// Inlined due to CJS import issues

export { createParser } from "./parse.js";
export type {
  EventSourceParseCallback,
  EventSourceParser,
  ParsedEvent,
  ParseEvent,
  ReconnectInterval,
} from "./types.js";
