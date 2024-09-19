import pRetry from "p-retry";
import PQueueMod from "p-queue";

const STATUS_NO_RETRY = [
  400, // Bad Request
  401, // Unauthorized
  403, // Forbidden
  404, // Not Found
  405, // Method Not Allowed
  406, // Not Acceptable
  407, // Proxy Authentication Required
  408, // Request Timeout
  422, // Unprocessable Entity
];
const STATUS_IGNORE = [
  409, // Conflict
];

type ResponseCallback = (response?: Response) => Promise<boolean>;

export interface AsyncCallerParams {
  /**
   * The maximum number of concurrent calls that can be made.
   * Defaults to `Infinity`, which means no limit.
   */
  maxConcurrency?: number;
  /**
   * The maximum number of retries that can be made for a single call,
   * with an exponential backoff between each attempt. Defaults to 6.
   */
  maxRetries?: number;

  onFailedResponseHook?: ResponseCallback;

  /**
   * Specify a custom fetch implementation.
   *
   * By default we expect the `fetch` is available in the global scope.
   */
  fetch?: typeof fetch | ((...args: any[]) => any);
}

export interface AsyncCallerCallOptions {
  signal?: AbortSignal;
}

/**
 * Do not rely on globalThis.Response, rather just
 * do duck typing
 */
function isResponse(x: unknown): x is Response {
  if (x == null || typeof x !== "object") return false;
  return "status" in x && "statusText" in x && "text" in x;
}

/**
 * Utility error to properly handle failed requests
 */
class HTTPError extends Error {
  status: number;
  text: string;

  response?: Response;

  constructor(status: number, message: string, response?: Response) {
    super(`HTTP ${status}: ${message}`);
    this.status = status;
    this.text = message;
    this.response = response;
  }

  static async fromResponse(
    response: Response,
    options?: { includeResponse?: boolean },
  ): Promise<HTTPError> {
    try {
      return new HTTPError(
        response.status,
        await response.text(),
        options?.includeResponse ? response : undefined,
      );
    } catch {
      return new HTTPError(
        response.status,
        response.statusText,
        options?.includeResponse ? response : undefined,
      );
    }
  }
}

/**
 * A class that can be used to make async calls with concurrency and retry logic.
 *
 * This is useful for making calls to any kind of "expensive" external resource,
 * be it because it's rate-limited, subject to network issues, etc.
 *
 * Concurrent calls are limited by the `maxConcurrency` parameter, which defaults
 * to `Infinity`. This means that by default, all calls will be made in parallel.
 *
 * Retries are limited by the `maxRetries` parameter, which defaults to 5. This
 * means that by default, each call will be retried up to 5 times, with an
 * exponential backoff between each attempt.
 */
export class AsyncCaller {
  protected maxConcurrency: AsyncCallerParams["maxConcurrency"];

  protected maxRetries: AsyncCallerParams["maxRetries"];

  private queue: (typeof import("p-queue"))["default"]["prototype"];

  private onFailedResponseHook?: ResponseCallback;

  private customFetch?: typeof fetch;

  constructor(params: AsyncCallerParams) {
    this.maxConcurrency = params.maxConcurrency ?? Infinity;
    this.maxRetries = params.maxRetries ?? 4;

    if ("default" in PQueueMod) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.queue = new (PQueueMod.default as any)({
        concurrency: this.maxConcurrency,
      });
    } else {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.queue = new (PQueueMod as any)({ concurrency: this.maxConcurrency });
    }
    this.onFailedResponseHook = params?.onFailedResponseHook;
    this.customFetch = params.fetch;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  call<A extends any[], T extends (...args: A) => Promise<any>>(
    callable: T,
    ...args: Parameters<T>
  ): Promise<Awaited<ReturnType<T>>> {
    const onFailedResponseHook = this.onFailedResponseHook;
    return this.queue.add(
      () =>
        pRetry(
          () =>
            callable(...(args as Parameters<T>)).catch(async (error) => {
              // eslint-disable-next-line no-instanceof/no-instanceof
              if (error instanceof Error) {
                throw error;
              } else if (isResponse(error)) {
                throw await HTTPError.fromResponse(error, {
                  includeResponse: !!onFailedResponseHook,
                });
              } else {
                throw new Error(error);
              }
            }),
          {
            async onFailedAttempt(error) {
              if (
                error.message.startsWith("Cancel") ||
                error.message.startsWith("TimeoutError") ||
                error.message.startsWith("AbortError")
              ) {
                throw error;
              }
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              if ((error as any)?.code === "ECONNABORTED") {
                throw error;
              }

              if (error instanceof HTTPError) {
                if (STATUS_NO_RETRY.includes(error.status)) {
                  throw error;
                } else if (STATUS_IGNORE.includes(error.status)) {
                  return;
                }
                if (onFailedResponseHook && error.response) {
                  await onFailedResponseHook(error.response);
                }
              }
            },
            // If needed we can change some of the defaults here,
            // but they're quite sensible.
            retries: this.maxRetries,
            randomize: true,
          },
        ),
      { throwOnTimeout: true },
    );
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  callWithOptions<A extends any[], T extends (...args: A) => Promise<any>>(
    options: AsyncCallerCallOptions,
    callable: T,
    ...args: Parameters<T>
  ): Promise<Awaited<ReturnType<T>>> {
    // Note this doesn't cancel the underlying request,
    // when available prefer to use the signal option of the underlying call
    if (options.signal) {
      return Promise.race([
        this.call<A, T>(callable, ...args),
        new Promise<never>((_, reject) => {
          options.signal?.addEventListener("abort", () => {
            reject(new Error("AbortError"));
          });
        }),
      ]);
    }
    return this.call<A, T>(callable, ...args);
  }

  fetch(...args: Parameters<typeof fetch>): ReturnType<typeof fetch> {
    const fetchFn = this.customFetch ?? fetch;
    return this.call(() =>
      fetchFn(...args).then((res) => (res.ok ? res : Promise.reject(res))),
    );
  }
}
