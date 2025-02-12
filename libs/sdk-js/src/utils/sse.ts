const mergeArrays = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const mergedArray = new Uint8Array(a.length + b.length);
  mergedArray.set(a);
  mergedArray.set(b, a.length);
  return mergedArray;
};

const CR = "\r".charCodeAt(0);
const LF = "\n".charCodeAt(0);
const NULL = "\0".charCodeAt(0);
const COLON = ":".charCodeAt(0);
const SPACE = " ".charCodeAt(0);

const TRAILING_NEWLINE = [CR, LF];

export class BytesLineDecoder extends TransformStream<Uint8Array, Uint8Array> {
  constructor() {
    let buffer = new Uint8Array();
    let trailingCr = false;

    super({
      start() {
        buffer = new Uint8Array();
        trailingCr = false;
      },

      transform(chunk, controller) {
        // See https://docs.python.org/3/glossary.html#term-universal-newlines
        let text = chunk;

        // Handle trailing CR from previous chunk
        if (trailingCr) {
          text = mergeArrays([CR], text);
          trailingCr = false;
        }

        // Check for trailing CR in current chunk
        if (text.length > 0 && text.at(-1) === CR) {
          trailingCr = true;
          text = text.subarray(0, -1);
        }

        if (!text.length) return;
        const trailingNewline = TRAILING_NEWLINE.includes(text.at(-1)!);

        // Pre-allocate lines array with estimated capacity
        let lines: Uint8Array[] = [];

        for (let offset = 0; offset < text.byteLength; ) {
          let idx = text.indexOf(CR, offset);
          if (idx === -1) idx = text.indexOf(LF, offset);
          if (idx === -1) {
            lines.push(text.subarray(offset));
            break;
          }

          lines.push(text.subarray(offset, idx));
          if (text[idx] === CR && text[idx + 1] === LF) {
            offset = idx + 2;
          } else {
            offset = idx + 1;
          }
        }

        if (lines.length === 1 && !trailingNewline) {
          buffer = mergeArrays(buffer, lines[0]);
          return;
        }

        if (buffer.length) {
          // Include existing buffer in first line
          buffer = mergeArrays(buffer, lines[0]);

          lines = lines.slice(1);
          lines.unshift(buffer);

          buffer = new Uint8Array();
        }

        if (!trailingNewline) {
          // If the last segment is not newline terminated,
          // buffer it for the next chunk
          if (lines.length) buffer = lines.pop()!;
        }

        // Enqueue complete lines
        for (const line of lines) {
          controller.enqueue(line);
        }
      },

      flush(controller) {
        if (buffer.length) {
          controller.enqueue(buffer);
        }
      },
    });
  }
}

interface StreamPart {
  event: string;
  data: unknown;
}

export class SSEDecoder extends TransformStream<Uint8Array, StreamPart> {
  constructor() {
    let event = "";
    let data: Uint8Array = new Uint8Array();
    let lastEventId = "";
    let retry: number | null = null;

    const decoder = new TextDecoder();

    super({
      transform(chunk, controller) {
        // Handle empty line case
        if (!chunk.length) {
          if (!event && !data.length && !lastEventId && retry == null) return;

          const sse = {
            event,
            data: data.length ? JSON.parse(decoder.decode(data)) : null,
          };

          // NOTE: as per the SSE spec, do not reset lastEventId
          event = "";
          data = new Uint8Array();
          retry = null;

          controller.enqueue(sse);
          return;
        }

        // Ignore comments
        if (chunk[0] === COLON) return;

        const sepIdx = chunk.indexOf(COLON);
        if (sepIdx === -1) return;

        const fieldName = decoder.decode(chunk.subarray(0, sepIdx));
        let value = chunk.subarray(sepIdx + 1);
        if (value[0] === SPACE) value = value.subarray(1);

        if (fieldName === "event") {
          event = decoder.decode(value);
        } else if (fieldName === "data") {
          data = mergeArrays(data, value);
        } else if (fieldName === "id") {
          if (value.indexOf(NULL) === -1) lastEventId = decoder.decode(value);
        } else if (fieldName === "retry") {
          const retryNum = Number.parseInt(decoder.decode(value));
          if (!Number.isNaN(retryNum)) retry = retryNum;
        }
      },

      flush(controller) {
        if (event) {
          controller.enqueue({
            event,
            data: data.length ? JSON.parse(decoder.decode(data)) : null,
          });
        }
      },
    });
  }
}
