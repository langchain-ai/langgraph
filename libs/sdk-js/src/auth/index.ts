type Maybe<T> = T | null | undefined;

interface AssistantConfig {
  tags?: Maybe<string[]>;
  recursion_limit?: Maybe<number>;
  configurable?: Maybe<{
    thread_id?: Maybe<string>;
    thread_ts?: Maybe<string>;
    [key: string]: unknown;
  }>;
}

interface AssistantCreate {
  assistant_id?: Maybe<string>;
  metadata?: Maybe<Record<string, unknown>>;
  config?: Maybe<AssistantConfig>;
  if_exists?: Maybe<"raise" | "do_nothing">;
  name?: Maybe<string>;
  graph_id: string;
}

interface AssistantRead {
  assistant_id: string;
  metadata?: Maybe<Record<string, unknown>>;
}
interface AssistantUpdate {
  assistant_id: string;
  metadata?: Maybe<Record<string, unknown>>;
  config?: Maybe<AssistantConfig>;
  graph_id?: Maybe<string>;
  name?: Maybe<string>;
  version?: Maybe<number>;
}
interface AssistantDelete {
  assistant_id: string;
}
interface AssistantSearch {
  graph_id?: Maybe<string>;
  metadata?: Maybe<Record<string, unknown>>;
  limit?: Maybe<number>;
  offset?: Maybe<number>;
}

interface ThreadCreate {
  thread_id?: Maybe<string>;
  metadata?: Maybe<Record<string, unknown>>;
  if_exists?: Maybe<"raise" | "do_nothing">;
}

interface ThreadRead {
  thread_id?: Maybe<string>;
}

interface ThreadUpdate {
  thread_id?: Maybe<string>;
  metadata?: Maybe<Record<string, unknown>>;
  action?: Maybe<"interrupt" | "rollback">;
}

interface ThreadDelete {
  thread_id?: Maybe<string>;
  run_id?: Maybe<string>;
}

interface ThreadSearch {
  thread_id?: Maybe<string>;
  status?: Maybe<"idle" | "busy" | "interrupted" | "error" | (string & {})>;
  metadata?: Maybe<Record<string, unknown>>;
  values?: Maybe<Record<string, unknown>>;
  limit?: Maybe<number>;
  offset?: Maybe<number>;
}

interface CronCreate {
  payload?: Maybe<Record<string, unknown>>;
  schedule: string;
  cron_id?: Maybe<string>;
  thread_id?: Maybe<string>;
  user_id?: Maybe<string>;
  end_time?: Maybe<string>;
}

interface CronRead {
  cron_id: string;
}

interface CronUpdate {
  cron_id: string;
  payload?: Maybe<Record<string, unknown>>;
  schedule?: Maybe<string>;
}

interface CronDelete {
  cron_id: string;
}

interface CronSearch {
  assistant_id?: Maybe<string>;
  thread_id?: Maybe<string>;
  limit?: Maybe<number>;
  offset?: Maybe<number>;
}

interface StorePut {
  namespace: string[];
  key: string;
  value: Record<string, unknown>;
}

interface StoreGet {
  namespace: Maybe<string[]>;
  key: string;
}

interface StoreSearch {
  namespace?: Maybe<string[]>;
  filter?: Maybe<Record<string, unknown>>;
  limit?: Maybe<number>;
  offset?: Maybe<number>;
  query?: Maybe<string>;
}

interface StoreListNamespaces {
  namespace?: Maybe<string[]>;
  suffix?: Maybe<string[]>;
  max_depth?: Maybe<number>;
  limit?: Maybe<number>;
  offset?: Maybe<number>;
}

interface StoreDelete {
  namespace?: Maybe<string[]>;
  key: string;
}

interface RunsCreate {
  thread_id?: Maybe<string>;
  assistant_id: string;
  run_id: string;
  status: Maybe<
    "pending" | "running" | "error" | "success" | "timeout" | "interrupted"
  >;
  metadata?: Maybe<Record<string, unknown>>;
  prevent_insert_if_inflight?: Maybe<boolean>;
  multitask_strategy?: Maybe<"interrupt" | "rollback" | "reject" | "enqueue">;
  if_not_exists?: Maybe<"reject" | "create">;
  after_seconds?: Maybe<number>;
  kwargs: Record<string, unknown>;
}

export interface ResourceActionType {
  ["threads:create"]: ThreadCreate;
  ["threads:read"]: ThreadRead;
  ["threads:update"]: ThreadUpdate;
  ["threads:delete"]: ThreadDelete;
  ["threads:search"]: ThreadSearch;
  ["threads:create_run"]: RunsCreate;

  ["assistants:create"]: AssistantCreate;
  ["assistants:read"]: AssistantRead;
  ["assistants:update"]: AssistantUpdate;
  ["assistants:delete"]: AssistantDelete;
  ["assistants:search"]: AssistantSearch;

  ["crons:create"]: CronCreate;
  ["crons:read"]: CronRead;
  ["crons:update"]: CronUpdate;
  ["crons:delete"]: CronDelete;
  ["crons:search"]: CronSearch;

  ["store:put"]: StorePut;
  ["store:get"]: StoreGet;
  ["store:search"]: StoreSearch;
  ["store:list_namespaces"]: StoreListNamespaces;
  ["store:delete"]: StoreDelete;
}

interface ResourceType {
  threads:
    | "threads:create"
    | "threads:read"
    | "threads:update"
    | "threads:delete"
    | "threads:search"
    | "threads:create_run";

  assistants:
    | "assistants:create"
    | "assistants:read"
    | "assistants:update"
    | "assistants:delete"
    | "assistants:search";
  crons:
    | "crons:create"
    | "crons:read"
    | "crons:update"
    | "crons:delete"
    | "crons:search";

  store:
    | "store:put"
    | "store:get"
    | "store:search"
    | "store:list_namespaces"
    | "store:delete";
}

interface ActionType {
  "*:create": "threads:create" | "assistants:create" | "crons:create";

  "*:read": "threads:read" | "assistants:read" | "crons:read";

  "*:update": "threads:update" | "assistants:update" | "crons:update";

  "*:delete":
    | "threads:delete"
    | "assistants:delete"
    | "crons:delete"
    | "store:delete";

  "*:search":
    | "threads:search"
    | "assistants:search"
    | "crons:search"
    | "store:search";

  "*:create_run": "threads:create_run";

  "*:put": "store:put";

  "*:get": "store:get";

  "*:list_namespaces": "store:list_namespaces";
}

interface BaseAuthContext {
  permissions?: string[];
  user?: {
    is_authenticated: boolean;
    display_name: string;
    identity: string;
    permissions: string[];
  };
}

type ContextMap = {
  [ActionType in keyof ResourceActionType]: {
    resource: ActionType extends `${infer Resource}:${string}`
      ? Resource
      : never;
    action: ActionType;
    data: ResourceActionType[ActionType];
    context: BaseAuthContext;
  };
};

type ActionCallbackParameter<
  T extends keyof ActionType,
  AuthContext = {},
> = ContextMap[ActionType[T]] & { context: AuthContext };

type AuthCallbackParameter<
  T extends keyof ResourceActionType,
  AuthContext = {},
> = ContextMap[T] & { context: AuthContext };

type ResourceCallbackParameter<
  T extends keyof ResourceType,
  AuthContext = {},
> = ContextMap[ResourceType[T]] & { context: AuthContext };

export type Filters<TKey extends string | number | symbol = string> = {
  [key in TKey]: string | { [op in "$contains" | "$eq"]?: string };
};

interface AuthenticateCallback<AuthContext extends Record<string, unknown>> {
  (request: Request): AuthContext;
}

interface GlobalCallback<
  Metadata extends Record<string, unknown> = {},
  AuthContext extends Record<string, unknown> = {},
> {
  (
    data: AuthCallbackParameter<keyof ResourceActionType, AuthContext>,
  ): void | boolean | Filters<keyof Metadata>;
}

interface ResourceCallback<
  Metadata extends Record<string, unknown> = {},
  AuthContext extends Record<string, unknown> = {},
> {
  (
    data: ResourceCallbackParameter<keyof ResourceType, AuthContext>,
  ): void | boolean | Filters<keyof Metadata>;
}

interface ActionCallback<
  Metadata extends Record<string, unknown> = {},
  AuthContext extends Record<string, unknown> = {},
> {
  (
    data: ActionCallbackParameter<keyof ActionType, AuthContext>,
  ): void | boolean | Filters<keyof Metadata>;
}

interface ResourceActionCallback<
  Metadata extends Record<string, unknown> = {},
  AuthContext extends Record<string, unknown> = {},
> {
  (
    data: AuthCallbackParameter<keyof ResourceActionType, AuthContext>,
  ): void | boolean | Filters<keyof Metadata>;
}

export class Auth<
  Metadata extends Record<string, unknown> = {},
  AuthContext extends Record<string, unknown> = {},
> {
  "~handlerCache": {
    authenticate?: AuthenticateCallback<AuthContext>;
    callbacks?: Record<
      string,
      (request: any) => void | boolean | Filters<keyof Metadata>
    >;
  } = {};

  authenticate(cb: AuthenticateCallback<AuthContext>): this {
    this["~handlerCache"].authenticate = cb;
    return this;
  }

  /**
   * Global handler for all requests
   */
  on(event: "*", callback: GlobalCallback): this;

  /**
   * Resource-specific handler
   */
  on<T extends keyof ResourceType>(
    event: T,
    callback: ResourceCallback<Metadata, AuthContext>,
  ): this;

  /**
   * Action-specific handler
   */
  on<T extends keyof ActionType>(
    event: T,
    callback: ActionCallback<Metadata, AuthContext>,
  ): this;

  /**
   * Resource-action specific handler
   */
  on<T extends keyof ResourceActionType>(
    event: T,
    callback: ResourceActionCallback,
  ): this;

  on(
    event: string,
    callback: (
      data: AuthCallbackParameter<keyof ResourceActionType, AuthContext>,
    ) => void | boolean | Filters<keyof Metadata>,
  ): this {
    this["~handlerCache"].callbacks ??= {};
    this["~handlerCache"].callbacks[event] = callback;
    return this;
  }
}
