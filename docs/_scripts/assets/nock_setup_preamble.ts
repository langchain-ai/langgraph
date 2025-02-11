import nock, { Definition } from "nock";
import msgpack from "msgpack-lite";
import zlib from "node:zlib";
import fs from "node:fs/promises";
import { Buffer } from "node:buffer";

// deno style imports here because we're running this in the deno jupyter kernel

interface NockCassetteData {
  hash: string;
  entries: Definition[];
}

// Utility functions for compression & serialization
function compressData(data: NockCassetteData, compressionLevel = 9): string {
  const packed = msgpack.encode(data);
  const compressed = zlib.deflateSync(packed, { level: compressionLevel });
  return compressed.toString("base64");
}

function decompressData(compressedString: string): NockCassetteData {
  const decoded = Buffer.from(compressedString, "base64");
  const decompressed = zlib.inflateSync(decoded);
  return msgpack.decode(decompressed) as NockCassetteData;
}

// deno-lint-ignore no-unused-vars
class HashedCassette {
  private recording = true;

  constructor(
    private readonly cassettePath: string,
    private readonly hash: string
  ) {}

  async enter() {
    try {
      const rawCassette = await fs.readFile(this.cassettePath, "utf-8");
      const data = decompressData(rawCassette);
      if (data.hash === this.hash) {
        this.recording = false;
        nock.disableNetConnect();
        nock.define(data.entries);
        return;
      }
    } catch (error) {
      if (error instanceof Error && error.message.includes("ENOENT")) {
        this.recording = true;
      } else {
        throw error;
      }
    }

    nock.recorder.rec({
      dont_print: true,
      output_objects: true,
    });
  }

  async exit() {
    if (this.recording) {
      const entries = nock.recorder.play() as Definition[];
      const data = {
        hash: this.hash,
        entries,
      };
      const compressed = compressData(data);
      await fs.writeFile(this.cassettePath, compressed);
    } else {
      nock.enableNetConnect();
      nock.restore();
      nock.cleanAll();
    }
  }
}
