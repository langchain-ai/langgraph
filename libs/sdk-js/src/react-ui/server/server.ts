import { v4 as uuidv4 } from "uuid";
import type { ComponentPropsWithoutRef, ElementType } from "react";
import type { RemoveUIMessage, UIMessage } from "../types.js";

export const typedUi = <Decl extends Record<string, ElementType>>(config: {
  writer?: (chunk: unknown) => void;
  runId?: string;
  metadata?: Record<string, unknown>;
  tags?: string[];
  runName?: string;
}) => {
  type PropMap = { [K in keyof Decl]: ComponentPropsWithoutRef<Decl[K]> };
  let collect: (UIMessage | RemoveUIMessage)[] = [];

  const runId = (config.metadata?.run_id as string | undefined) ?? config.runId;
  if (!runId) throw new Error("run_id is required");

  const metadata = {
    ...config.metadata,
    tags: config.tags,
    name: config.runName,
    run_id: runId,
  };

  const create = <K extends keyof PropMap & string>(
    name: K,
    props: PropMap[K],
  ): UIMessage => ({
    type: "ui" as const,
    id: uuidv4(),
    name,
    content: props,
    additional_kwargs: metadata,
  });

  const remove = (id: string): RemoveUIMessage => ({ type: "remove-ui", id });

  return {
    create,
    remove,

    collect,
    write: <K extends keyof PropMap & string>(name: K, props: PropMap[K]) => {
      const evt: UIMessage = create(name, props);
      collect.push(evt);
      config.writer?.(evt);
    },
  };
};
