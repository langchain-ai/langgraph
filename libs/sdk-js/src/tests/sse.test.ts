import { Readable } from "node:stream";
import { IterableReadableStream } from "../utils/stream.js";
import { BytesLineDecoder, SSEDecoder } from "../utils/sse.js";

describe("BytesLineDecoder", () => {
  const createStream = (chunks: Uint8Array[]) => {
    return Readable.toWeb(Readable.from(chunks)) as ReadableStream<Uint8Array>;
  };

  const gather = async (
    stream: ReadableStream<Uint8Array>,
  ): Promise<Uint8Array[]> => {
    const results: Uint8Array[] = [];
    for await (const chunk of IterableReadableStream.fromReadableStream(
      stream,
    )) {
      results.push(chunk);
    }
    return results;
  };

  const textEncoder = new TextEncoder();
  const textDecoder = new TextDecoder();

  test("handles single line with newline", async () => {
    const input = createStream([textEncoder.encode("hello\n")]);
    const decoded = input.pipeThrough(new BytesLineDecoder());
    const results = await gather(decoded);

    expect(results.length).toBe(1);
    expect(textDecoder.decode(results[0])).toBe("hello");
  });

  test("handles multiple lines", async () => {
    const input = createStream([textEncoder.encode("line1\nline2\nline3\n")]);
    const decoded = input.pipeThrough(new BytesLineDecoder());
    const results = await gather(decoded);

    expect(results.length).toBe(3);
    expect(textDecoder.decode(results[0])).toBe("line1");
    expect(textDecoder.decode(results[1])).toBe("line2");
    expect(textDecoder.decode(results[2])).toBe("line3");
  });

  test("handles split chunks", async () => {
    const input = createStream([
      textEncoder.encode("li"),
      textEncoder.encode("ne1\nli"),
      textEncoder.encode("ne2\n"),
    ]);
    const decoded = input.pipeThrough(new BytesLineDecoder());
    const results = await gather(decoded);

    expect(results.length).toBe(2);
    expect(textDecoder.decode(results[0])).toBe("line1");
    expect(textDecoder.decode(results[1])).toBe("line2");
  });

  test("handles CR LF line endings", async () => {
    const input = createStream([textEncoder.encode("line1\r\nline2\r\n")]);
    const decoded = input.pipeThrough(new BytesLineDecoder());
    const results = await gather(decoded);

    expect(results.length).toBe(2);
    expect(textDecoder.decode(results[0])).toBe("line1");
    expect(textDecoder.decode(results[1])).toBe("line2");
  });

  test("handles split CR LF", async () => {
    const input = createStream([
      textEncoder.encode("line1\r"),
      textEncoder.encode("\nline2\r\n"),
    ]);
    const decoded = input.pipeThrough(new BytesLineDecoder());
    const results = await gather(decoded);

    expect(results.length).toBe(2);
    expect(textDecoder.decode(results[0])).toBe("line1");
    expect(textDecoder.decode(results[1])).toBe("line2");
  });
});

describe("SSEDecoder", () => {
  const createStream = (lines: string[]) => {
    const encoder = new TextEncoder();
    const chunks = lines.map((line) => encoder.encode(line));
    return Readable.toWeb(Readable.from(chunks)) as ReadableStream<Uint8Array>;
  };

  const collectResults = async (
    stream: ReadableStream<any>,
  ): Promise<any[]> => {
    const results: any[] = [];
    for await (const chunk of IterableReadableStream.fromReadableStream(
      stream,
    )) {
      results.push(chunk);
    }
    return results;
  };

  test("decodes simple event", async () => {
    const input = createStream([
      "event: test\n",
      'data: {"message": "hello"}\n',
      "\n",
    ]);
    const decoded = input
      .pipeThrough(new BytesLineDecoder())
      .pipeThrough(new SSEDecoder());

    const results = await collectResults(decoded);
    expect(results.length).toBe(1);
    expect(results[0]).toEqual({
      event: "test",
      data: { message: "hello" },
    });
  });

  test("ignores comments", async () => {
    const input = createStream([
      ": this is a comment\n",
      "event: test\n",
      'data: {"message": "hello"}\n',
    ]);
    const decoded = input
      .pipeThrough(new BytesLineDecoder())
      .pipeThrough(new SSEDecoder());

    const results = await collectResults(decoded);
    expect(results.length).toBe(1);
    expect(results[0]).toEqual({
      event: "test",
      data: { message: "hello" },
    });
  });

  test("handles multiple events", async () => {
    const input = createStream([
      "event: test1\n",
      'data: {"message": "hello"}\n',
      "\n",
      "event: test2\n",
      'data: {"message": "world"}\n',
      "\n",
    ]);
    const decoded = input
      .pipeThrough(new BytesLineDecoder())
      .pipeThrough(new SSEDecoder());

    const results = await collectResults(decoded);
    expect(results.length).toBe(2);
    expect(results[0]).toEqual({
      event: "test1",
      data: { message: "hello" },
    });
    expect(results[1]).toEqual({
      event: "test2",
      data: { message: "world" },
    });
  });

  test("end event without data", async () => {
    const input = createStream(["event: test\n"]);
    const decoded = input
      .pipeThrough(new BytesLineDecoder())
      .pipeThrough(new SSEDecoder());

    const results = await collectResults(decoded);
    expect(results.length).toBe(1);
    expect(results[0]).toEqual({
      event: "test",
      data: null,
    });
  });

  test("end event without newline", async () => {
    const input = createStream(["event: end"]);
    const decoded = input
      .pipeThrough(new BytesLineDecoder())
      .pipeThrough(new SSEDecoder());

    const results = await collectResults(decoded);
    expect(results.length).toBe(1);
    expect(results[0]).toEqual({
      event: "end",
      data: null,
    });
  });
});
