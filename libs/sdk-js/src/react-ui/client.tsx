"use client";

import { useStream } from "../react/index.js";
import type { UIMessage } from "./types.js";

import * as React from "react";
import * as ReactDOM from "react-dom";
import * as JsxRuntime from "react/jsx-runtime";
import type { UseStream } from "../react/stream.js";

const UseStreamContext = React.createContext<{
  stream: ReturnType<typeof useStream>;
  meta: unknown;
}>(null!);

type BagTemplate = {
  ConfigurableType?: Record<string, unknown>;
  InterruptType?: unknown;
  CustomEventType?: unknown;
  UpdateType?: unknown;
  MetaType?: unknown;
};

type GetMetaType<Bag extends BagTemplate> = Bag extends { MetaType: unknown }
  ? Bag["MetaType"]
  : unknown;

interface UseStreamContext<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  Bag extends BagTemplate = BagTemplate,
> extends UseStream<StateType, Bag> {
  meta?: GetMetaType<Bag>;
}

export function useStreamContext<
  StateType extends Record<string, unknown> = Record<string, unknown>,
  Bag extends {
    ConfigurableType?: Record<string, unknown>;
    InterruptType?: unknown;
    CustomEventType?: unknown;
    UpdateType?: unknown;
    MetaType?: unknown;
  } = BagTemplate,
>(): UseStreamContext<StateType, Bag> {
  const ctx = React.useContext(UseStreamContext);
  if (!ctx) {
    throw new Error(
      "useStreamContext must be used within a LoadExternalComponent",
    );
  }

  return new Proxy(ctx, {
    get(target, prop: keyof UseStreamContext<StateType, Bag>) {
      if (prop === "meta") return target.meta;
      return target.stream[prop];
    },
  }) as unknown as UseStreamContext<StateType, Bag>;
}

interface ComponentTarget {
  comp: React.FunctionComponent | React.ComponentClass;
  target: HTMLElement;
}

class ComponentStore {
  private cache: Record<string, ComponentTarget> = {};
  private boundCache: Record<
    string,
    {
      subscribe: (onStoreChange: () => void) => () => void;
      getSnapshot: () => ComponentTarget | undefined;
    }
  > = {};
  private callbacks: Record<
    string,
    ((
      comp: React.FunctionComponent | React.ComponentClass,
      el: HTMLElement,
    ) => void)[]
  > = {};

  respond(
    shadowRootId: string,
    comp: React.FunctionComponent | React.ComponentClass,
    targetElement: HTMLElement,
  ) {
    this.cache[shadowRootId] = { comp, target: targetElement };
    this.callbacks[shadowRootId]?.forEach((c) => c(comp, targetElement));
  }

  getBoundStore(shadowRootId: string) {
    this.boundCache[shadowRootId] ??= {
      subscribe: (onStoreChange: () => void) => {
        this.callbacks[shadowRootId] ??= [];
        this.callbacks[shadowRootId].push(onStoreChange);
        return () => {
          this.callbacks[shadowRootId] = this.callbacks[shadowRootId].filter(
            (c) => c !== onStoreChange,
          );
        };
      },
      getSnapshot: () => this.cache[shadowRootId],
    };

    return this.boundCache[shadowRootId];
  }
}

const COMPONENT_STORE = new ComponentStore();
const EXT_STORE_SYMBOL = Symbol.for("LGUI_EXT_STORE");
const REQUIRE_SYMBOL = Symbol.for("LGUI_REQUIRE");

interface LoadExternalComponentProps
  extends Pick<React.HTMLAttributes<HTMLDivElement>, "style" | "className"> {
  /** Stream of the assistant */
  stream: ReturnType<typeof useStream>;

  /** UI message to be rendered */
  message: UIMessage;

  /** Additional context to be passed to the child component */
  meta?: unknown;

  /** Fallback to be rendered when the component is loading */
  fallback?: React.ReactNode;

  /**
   * Map of components that can be rendered directly without fetching the UI code
   * from the server.
   */
  components?: Record<string, React.FunctionComponent | React.ComponentClass>;
}

export function LoadExternalComponent({
  stream,
  message,
  meta,
  fallback,
  components,
  ...props
}: LoadExternalComponentProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  const shadowRootId = `child-shadow-${id}`;

  const store = React.useMemo(
    () => COMPONENT_STORE.getBoundStore(shadowRootId),
    [shadowRootId],
  );
  const state = React.useSyncExternalStore(store.subscribe, store.getSnapshot);

  const clientComponent = components?.[message.name];
  const hasClientComponent = clientComponent != null;

  const uiClient = stream.client["~ui"];
  React.useEffect(() => {
    if (hasClientComponent) return;
    uiClient.getComponent(stream.assistantId, message.name).then((html) => {
      const dom = ref.current;
      if (!dom) return;
      const root = dom.shadowRoot ?? dom.attachShadow({ mode: "open" });
      const fragment = document
        .createRange()
        .createContextualFragment(
          html.replace("{{shadowRootId}}", shadowRootId),
        );
      root.appendChild(fragment);
    });
  }, [
    uiClient,
    stream.assistantId,
    message.name,
    shadowRootId,
    hasClientComponent,
  ]);

  if (hasClientComponent) {
    return React.createElement(clientComponent, message.props);
  }

  return (
    <>
      <div id={shadowRootId} ref={ref} {...props} />

      <UseStreamContext.Provider value={{ stream, meta }}>
        {state?.target != null
          ? ReactDOM.createPortal(
              React.createElement(state.comp, message.props),
              state.target,
            )
          : fallback}
      </UseStreamContext.Provider>
    </>
  );
}

declare global {
  interface Window {
    [EXT_STORE_SYMBOL]: ComponentStore;
    [REQUIRE_SYMBOL]: (name: string) => unknown;
  }
}

export function bootstrapUiContext() {
  if (typeof window === "undefined") {
    console.warn(
      "Attempting to bootstrap UI context outside of browser environment. " +
        "Avoid importing from `@langchain/langgraph-sdk/react-ui` in server context.",
    );
    return;
  }

  window[EXT_STORE_SYMBOL] = COMPONENT_STORE;
  window[REQUIRE_SYMBOL] = (name: string) => {
    if (name === "react") return React;
    if (name === "react-dom") return ReactDOM;
    if (name === "react/jsx-runtime") return JsxRuntime;
    if (name === "@langchain/langgraph-sdk/react") return { useStream };
    if (name === "@langchain/langgraph-sdk/react-ui") {
      return {
        useStreamContext,
        LoadExternalComponent: () => {
          throw new Error("Nesting LoadExternalComponent is not supported");
        },
      };
    }

    throw new Error(`Unknown module...: ${name}`);
  };
}
