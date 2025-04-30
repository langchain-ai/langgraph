import { v4 as uuidv4 } from "uuid";
import type { ComponentPropsWithoutRef, ElementType } from "react";
import type { RemoveUIMessage, UIMessage } from "../types.js";

interface MessageLike {
  id?: string;
}

/**
 * Helper to send and persist UI messages. Accepts a map of component names to React components
 * as type argument to provide type safety. Will also write to the `options?.stateKey` state.
 *
 * @param config LangGraphRunnableConfig
 * @param options
 * @returns
 */
export const typedUi = <Decl extends Record<string, ElementType>>(
  config: {
    writer?: (chunk: unknown) => void;
    runId?: string;
    metadata?: Record<string, unknown>;
    tags?: string[];
    runName?: string;
    configurable?: {
      __pregel_send?: (writes_: [string, unknown][]) => void;
      [key: string]: unknown;
    };
  },
  options?: {
    /** The key to write the UI messages to. Defaults to `ui`. */
    stateKey?: string;
  },
) => {
  type PropMap = { [K in keyof Decl]: ComponentPropsWithoutRef<Decl[K]> };
  let items: (UIMessage | RemoveUIMessage)[] = [];
  const stateKey = options?.stateKey ?? "ui";

  const runId = (config.metadata?.run_id as string | undefined) ?? config.runId;
  if (!runId) throw new Error("run_id is required");

  function handlePush<K extends keyof PropMap & string>(
    message: {
      id?: string;
      name: K;
      props: PropMap[K];
      metadata?: Record<string, unknown>;
    },
    options?: { message?: MessageLike; merge?: boolean },
  ): UIMessage<K, PropMap[K]>;

  function handlePush<K extends keyof PropMap & string>(
    message: {
      id?: string;
      name: K;
      props: Partial<PropMap[K]>;
      metadata?: Record<string, unknown>;
    },
    options: { message?: MessageLike; merge: true },
  ): UIMessage<K, Partial<PropMap[K]>>;

  function handlePush<K extends keyof PropMap & string>(
    message: {
      id?: string;
      name: K;
      props: PropMap[K] | Partial<PropMap[K]>;
      metadata?: Record<string, unknown>;
    },
    options?: { message?: MessageLike; merge?: boolean },
  ): UIMessage<K, PropMap[K] | Partial<PropMap[K]>> {
    const evt: UIMessage<K, PropMap[K] | Partial<PropMap[K]>> = {
      type: "ui" as const,
      id: message?.id ?? uuidv4(),
      name: message?.name,
      props: message?.props,
      metadata: {
        merge: options?.merge || undefined,
        run_id: runId,
        tags: config.tags,
        name: config.runName,
        ...message?.metadata,
        ...(options?.message ? { message_id: options.message.id } : null),
      },
    };
    items.push(evt);
    config.writer?.(evt);
    config.configurable?.__pregel_send?.([[stateKey, evt]]);
    return evt;
  }

  const handleDelete = (id: string): RemoveUIMessage => {
    const evt: RemoveUIMessage = { type: "remove-ui", id };
    items.push(evt);
    config.writer?.(evt);
    config.configurable?.__pregel_send?.([[stateKey, evt]]);
    return evt;
  };

  return { push: handlePush, delete: handleDelete, items };
};
