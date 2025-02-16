import type { useStream } from "../react/index.js";
import type { UIMessage } from "./types.js";

import * as React from "react";
import * as ReactDOM from "react-dom";
import * as JsxRuntime from "react/jsx-runtime";

const UseStreamContext = React.createContext<ReturnType<typeof useStream>>(
  null!,
);

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
    name: string,
    comp: React.FunctionComponent | React.ComponentClass,
    targetElement: HTMLElement,
  ) {
    this.cache[name] = { comp, target: targetElement };
    this.callbacks[name]?.forEach((c) => c(comp, targetElement));
  }

  getBoundStore(name: string) {
    this.boundCache[name] ??= {
      subscribe: (onStoreChange: () => void) => {
        this.callbacks[name] ??= [];
        this.callbacks[name].push(onStoreChange);
        return () => {
          this.callbacks[name] = this.callbacks[name].filter(
            (c) => c !== onStoreChange,
          );
        };
      },
      getSnapshot: () => this.cache[name],
    };

    return this.boundCache[name];
  }
}

const COMPONENT_STORE = new ComponentStore();
const EXT_STORE_SYMBOL = Symbol.for("LGUI_EXT_STORE");
const REQUIRE_SYMBOL = Symbol.for("LGUI_REQUIRE");

export function LoadExternalComponent({
  stream,
  message,
  className,
}: {
  stream: ReturnType<typeof useStream>;
  message: UIMessage;
  className?: string;
}) {
  const ref = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  const shadowRootId = `child-shadow-${id}`;

  const store = React.useMemo(
    () => COMPONENT_STORE.getBoundStore(shadowRootId),
    [shadowRootId],
  );
  const state = React.useSyncExternalStore(store.subscribe, store.getSnapshot);

  React.useEffect(() => {
    fetch("http://localhost:3123", {
      headers: { "Content-Type": "application/json" },
      method: "POST",
      body: JSON.stringify({ name: message.name, shadowRootId }),
    })
      .then((a) => a.text())
      .then((html) => {
        const dom = ref.current;
        if (!dom) return;
        const root = dom.shadowRoot ?? dom.attachShadow({ mode: "open" });
        const fragment = document.createRange().createContextualFragment(html);
        root.appendChild(fragment);
      });
  }, [message.name, shadowRootId]);

  return (
    <>
      <div id={shadowRootId} ref={ref} className={className} />

      <UseStreamContext.Provider value={stream}>
        {state?.target &&
          ReactDOM.createPortal(
            React.createElement(state.comp, message.content),
            state.target,
          )}
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

window[EXT_STORE_SYMBOL] = COMPONENT_STORE;
window[REQUIRE_SYMBOL] = (name: string) => {
  if (name === "react") return React;
  if (name === "react-dom") return ReactDOM;
  if (name === "react/jsx-runtime") return JsxRuntime;
  if (name === "@langchain/langgraph-sdk/react") {
    return { useStream: () => React.useContext(UseStreamContext) };
  }

  throw new Error(`Unknown module...: ${name}`);
};
