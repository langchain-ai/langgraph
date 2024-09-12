import type { JSONSchema7 } from "json-schema";

type Optional<T> = T | null | undefined;

type RunStatus =
  | "pending"
  | "running"
  | "error"
  | "success"
  | "timeout"
  | "interrupted";

type ThreadStatus = "idle" | "busy" | "interrupted";

export interface Config {
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
  configurable: {
    /**
     * ID of the thread
     */
    thread_id?: string;

    /**
     * Timestamp of the state checkpoint
     */
    checkpoint_id?: string;
    [key: string]: unknown;
  };
}

export interface GraphSchema {
  /**
   * The ID of the graph.
   */
  graph_id: string;

  /**
   * The schema for the input state.
   * Missing if unable to generate JSON schema from graph.
   */
  input_schema?: JSONSchema7;

  /**
   * The schema for the output state.
   * Missing if unable to generate JSON schema from graph.
   */
  output_schema?: JSONSchema7;

  /**
   * The schema for the graph state.
   * Missing if unable to generate JSON schema from graph.
   */
  state_schema?: JSONSchema7;

  /**
   * The schema for the graph config.
   * Missing if unable to generate JSON schema from graph.
   */
  config_schema?: JSONSchema7;
}

export type Metadata = Optional<Record<string, unknown>>;

export interface Assistant {
  assistant_id: string;
  graph_id: string;
  config: Config;
  created_at: string;
  updated_at: string;
  metadata: Metadata;
  version: number;
  assistant_name: string;
}
export type AssistantGraph = Record<string, Array<Record<string, unknown>>>;

export interface Thread {
  thread_id: string;
  created_at: string;
  updated_at: string;
  metadata: Metadata;
  status: ThreadStatus;
}

export interface Cron {
  cron_id: string;
  thread_id: Optional<string>;
  end_time: Optional<string>;
  schedule: string;
  created_at: string;
  updated_at: string;
  payload: Record<string, unknown>;
}

export type DefaultValues = Record<string, unknown>[] | Record<string, unknown>;

export interface ThreadState<ValuesType = DefaultValues> {
  values: ValuesType;
  next: string[];
  checkpoint_id: string;
  metadata: Metadata;
  created_at: Optional<string>;
  parent_checkpoint_id: Optional<string>;

  config: Config;
  parent_config?: Config;
}

export interface Run {
  run_id: string;
  thread_id: string;
  assistant_id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  metadata: Metadata;
}
