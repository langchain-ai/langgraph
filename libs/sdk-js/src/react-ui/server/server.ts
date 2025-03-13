import { v4 as uuidv4 } from "uuid";
import type { ComponentPropsWithoutRef, ElementType } from "react";
import type { RemoveUIMessage, UIMessage } from "../types.js";

interface MessageLike {
  id?: string;
}

export const typedUi = <Decl extends Record<string, ElementType>>(config: {
  writer?: (chunk: unknown) => void;
  runId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  runName?: string;
}) => {
  type PropMap = { [K in keyof Decl]: ComponentPropsWithoutRef<Decl[K]> };
  let items: (UIMessage | RemoveUIMessage)[] = [];

  const runId = (config.metadata?.run_id as string | undefined) ?? config.runId;
  if (!runId) throw new Error("run_id is required");

  const metadata = {
    ...config.metadata,
    tags: config.tags,
    name: config.runName,
    run_id: runId,
  };

  const handlePush = <K extends keyof PropMap & string>(
    message: {
      id?: string;
      name: K;
      props: PropMap[K];
      metadata?: Record<string, unknown>;
    },
    options?: { message?: MessageLike },
  ): UIMessage => {
    const evt: UIMessage = {
      type: "ui" as const,
      id: message?.id ?? uuidv4(),
      name: message?.name,
      props: message?.props,
      metadata: {
        ...metadata,
        ...message?.metadata,
        ...(options?.message ? { message_id: options.message.id } : null),
      },
    };
    items.push(evt);
    config.writer?.(evt);
    return evt;
  };

  const handleDelete = (id: string): RemoveUIMessage => {
    const evt: RemoveUIMessage = { type: "remove-ui", id };
    items.push(evt);
    config.writer?.(evt);
    return evt;
  };

  return { push: handlePush, delete: handleDelete, items };
};
