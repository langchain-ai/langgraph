/* eslint-disable no-process-env */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { jest } from "@jest/globals";
import { Client } from "../client.js";
import { overrideFetchImplementation } from "../singletons/fetch.js";

describe.each([[""], ["mocked"]])("Client uses %s fetch", (description) => {
  let globalFetchMock: jest.Mock;
  let overriddenFetch: jest.Mock;
  let expectedFetchMock: jest.Mock;
  let unexpectedFetchMock: jest.Mock;

  beforeEach(() => {
    globalFetchMock = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            batch_ingest_config: {
              use_multipart_endpoint: true,
            },
          }),
        text: () => Promise.resolve(""),
      }),
    );
    overriddenFetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            batch_ingest_config: {
              use_multipart_endpoint: true,
            },
          }),
        text: () => Promise.resolve(""),
      }),
    );
    expectedFetchMock =
      description === "mocked" ? overriddenFetch : globalFetchMock;
    unexpectedFetchMock =
      description === "mocked" ? globalFetchMock : overriddenFetch;

    if (description === "mocked") {
      overrideFetchImplementation(overriddenFetch);
    } else {
      overrideFetchImplementation(globalFetchMock);
    }
    // Mock global fetch
    (globalThis as any).fetch = globalFetchMock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("createRuns", () => {
    it("should create an example with the given input and generation", async () => {
      const client = new Client({ apiKey: "test-api-key" });

      const thread = await client.threads.create();
      expect(expectedFetchMock).toHaveBeenCalledTimes(1);
      expect(unexpectedFetchMock).not.toHaveBeenCalled();

      jest.clearAllMocks(); // Clear all mocks before the next operation

      // Then clear & run the function
      await client.runs.create(thread.thread_id, "somegraph", {
        input: { foo: "bar" },
      });
      expect(expectedFetchMock).toHaveBeenCalledTimes(1);
      expect(unexpectedFetchMock).not.toHaveBeenCalled();
    });
  });
});
