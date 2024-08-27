import { Config, Metadata } from "./schema.js";

export type StreamMode = "values" | "messages" | "updates" | "events" | "debug";
export type MultitaskStrategy = "reject" | "interrupt" | "rollback" | "enqueue";
export type OnConflictBehavior = "raise" | "do_nothing";
export type StreamEvent =
  | "events"
  | "metadata"
  | "debug"
  | "updates"
  | "values"
  | "messages/partial"
  | "messages/metadata"
  | "messages/complete"
  | (string & {});

interface RunsInvokePayload {
  /**
   * Input to the run. Pass `null` to resume from the current state of the thread.
   */
  input?: Record<string, unknown> | null;

  /**
   * Metadata for the run.
   */
  metadata?: Metadata;

  /**
   * Additional configuration for the run.
   */
  config?: Config;

  /**
   * Checkpoint ID for when creating a new run.
   */
  checkpointId?: string;

  /**
   * Interrupt execution before entering these nodes.
   */
  interruptBefore?: string[];

  /**
   * Interrupt execution after leaving these nodes.
   */
  interruptAfter?: string[];

  /**
   * Strategy to handle concurrent runs on the same thread. Only relevant if
   * there is a pending/inflight run on the same thread. One of:
   * - "reject": Reject the new run.
   * - "interrupt": Interrupt the current run, keeping steps completed until now,
       and start a new one.
   * - "rollback": Cancel and delete the existing run, rolling back the thread to
      the state before it had started, then start the new run.
   * - "enqueue": Queue up the new run to start after the current run finishes.
   */
  multitaskStrategy?: MultitaskStrategy;

  /**
   * Abort controller signal to cancel the run.
   */
  signal?: AbortController["signal"];
}

export interface RunsStreamPayload extends RunsInvokePayload {
  /**
   * One of `"values"`, `"messages"`, `"updates"` or `"events"`.
   * - `"values"`: Stream the thread state any time it changes.
   * - `"messages"`: Stream chat messages from thread state and calls to chat models,
   *                 token-by-token where possible.
   * - `"updates"`: Stream the state updates returned by each node.
   * - `"events"`: Stream all events produced by the run. You can also access these
   *               afterwards using the `client.runs.listEvents()` method.
   */
  streamMode?: StreamMode | Array<StreamMode>;

  /**
   * Pass one or more feedbackKeys if you want to request short-lived signed URLs
   * for submitting feedback to LangSmith with this key for this run.
   */
  feedbackKeys?: string[];
}

export interface RunsCreatePayload extends RunsInvokePayload {
  /**
   * Webhook to call when the run is complete.
   */
  webhook?: string;
}

export interface CronsCreatePayload extends RunsCreatePayload {
  /**
   * Schedule for running the Cron Job
   */
  schedule: string;
}

export type RunsWaitPayload = RunsStreamPayload;
