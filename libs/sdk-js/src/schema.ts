import type { JSONSchema7 } from "json-schema";

type Optional<T> = T | null | undefined;

export type RunStatus =
  | "pending"
  | "running"
  | "error"
  | "success"
  | "timeout"
  | "interrupted";

export type ThreadStatus = "idle" | "busy" | "interrupted" | "error";

type MultitaskStrategy = "reject" | "interrupt" | "rollback" | "enqueue";

export type CancelAction = "interrupt" | "rollback";

export type Config = {
  /**
   * Tags for this call and any sub-calls (eg. a Chain calling an LLM).
   * You can use these to filter calls.
   */
  tags?: string[];

  /**
   * Maximum number of times a call can recurse.
   * If not provided, defaults to 25.
   */
  recursion_limit?: number;

  /**
   * Runtime values for attributes previously made configurable on this Runnable.
   */
  configurable?: {
    /**
     * ID of the thread
     */
    thread_id?: Optional<string>;

    /**
     * Timestamp of the state checkpoint
     */
    checkpoint_id?: Optional<string>;

    [key: string]: unknown;
  };
};

export interface GraphSchema {
  /**
   * The ID of the graph.
   */
  graph_id: string;

  /**
   * The schema for the input state.
   * Missing if unable to generate JSON schema from graph.
   */
  input_schema?: JSONSchema7 | null | undefined;

  /**
   * The schema for the output state.
   * Missing if unable to generate JSON schema from graph.
   */
  output_schema?: JSONSchema7 | null | undefined;

  /**
   * The schema for the graph state.
   * Missing if unable to generate JSON schema from graph.
   */
  state_schema?: JSONSchema7 | null | undefined;

  /**
   * The schema for the graph config.
   * Missing if unable to generate JSON schema from graph.
   */
  config_schema?: JSONSchema7 | null | undefined;
}

export type Subgraphs = Record<string, GraphSchema>;

export type Metadata = Optional<{
  source?: "input" | "loop" | "update" | (string & {});

  step?: number;

  writes?: Record<string, unknown> | null;

  parents?: Record<string, string>;

  [key: string]: unknown;
}>;

export interface AssistantBase {
  /** The ID of the assistant. */
  assistant_id: string;

  /** The ID of the graph. */
  graph_id: string;

  /** The assistant config. */
  config: Config;

  /** The time the assistant was created. */
  created_at: string;

  /** The assistant metadata. */
  metadata: Metadata;

  /** The version of the assistant. */
  version: number;

  /** The name of the assistant */
  name: string;
}

export interface AssistantVersion extends AssistantBase {}

export interface Assistant extends AssistantBase {
  /** The last time the assistant was updated. */
  updated_at: string;
}

export interface AssistantGraph {
  nodes: Array<{
    id: string | number;
    name?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data?: Record<string, any> | string;
    metadata?: unknown;
  }>;
  edges: Array<{
    source: string;
    target: string;
    data?: string;
    conditional?: boolean;
  }>;
}

/**
 * An interrupt thrown inside a thread.
 */
export interface Interrupt<TValue = unknown> {
  value?: TValue;
  when: "during" | (string & {});
  resumable?: boolean;
  ns?: string[];
}

export interface Thread<ValuesType = DefaultValues> {
  /** The ID of the thread. */
  thread_id: string;

  /** The time the thread was created. */
  created_at: string;

  /** The last time the thread was updated. */
  updated_at: string;

  /** The thread metadata. */
  metadata: Metadata;

  /** The status of the thread */
  status: ThreadStatus;

  /** The current state of the thread. */
  values: ValuesType;

  /** Interrupts which were thrown in this thread */
  interrupts: Record<string, Array<Interrupt>>;
}

export interface Cron {
  /** The ID of the cron */
  cron_id: string;

  /** The ID of the thread */
  thread_id: Optional<string>;

  /** The end date to stop running the cron. */
  end_time: Optional<string>;

  /** The schedule to run, cron format. */
  schedule: string;

  /** The time the cron was created. */
  created_at: string;

  /** The last time the cron was updated. */
  updated_at: string;

  /** The run payload to use for creating new run. */
  payload: Record<string, unknown>;
}

export type DefaultValues = Record<string, unknown>[] | Record<string, unknown>;

export interface ThreadState<ValuesType = DefaultValues> {
  /** The state values */
  values: ValuesType;

  /** The next nodes to execute. If empty, the thread is done until new input is received */
  next: string[];

  /** Checkpoint of the thread state */
  checkpoint: Checkpoint;

  /** Metadata for this state */
  metadata: Metadata;

  /** Time of state creation  */
  created_at: Optional<string>;

  /** The parent checkpoint. If missing, this is the root checkpoint */
  parent_checkpoint: Optional<Checkpoint>;

  /** Tasks to execute in this step. If already attempted, may contain an error */
  tasks: Array<ThreadTask>;
}

export interface ThreadTask {
  id: string;
  name: string;
  result?: unknown;
  error: Optional<string>;
  interrupts: Array<Interrupt>;
  checkpoint: Optional<Checkpoint>;
  state: Optional<ThreadState>;
}

export interface Run {
  /** The ID of the run */
  run_id: string;

  /** The ID of the thread */
  thread_id: string;

  /** The assistant that wwas used for this run */
  assistant_id: string;

  /** The time the run was created */
  created_at: string;

  /** The last time the run was updated */
  updated_at: string;

  /** The status of the run. */
  status: RunStatus;

  /** Run metadata */
  metadata: Metadata;

  /** Strategy to handle concurrent runs on the same thread */
  multitask_strategy: Optional<MultitaskStrategy>;
}

export type Checkpoint = {
  thread_id: string;
  checkpoint_ns: string;
  checkpoint_id: Optional<string>;
  checkpoint_map: Optional<Record<string, unknown>>;
};

export interface ListNamespaceResponse {
  namespaces: string[][];
}
export interface Item {
  namespace: string[];
  key: string;
  value: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

export interface SearchItem extends Item {
  score?: number;
}
export interface SearchItemsResponse {
  items: SearchItem[];
}

export interface CronCreateResponse {
  cron_id: string;
  assistant_id: string;
  thread_id: string | undefined;
  user_id: string;
  payload: Record<string, unknown>;
  schedule: string;
  next_run_date: string;
  end_time: string | undefined;
  created_at: string;
  updated_at: string;
  metadata: Metadata;
}

export interface CronCreateForThreadResponse
  extends Omit<CronCreateResponse, "thread_id"> {
  thread_id: string;
}
