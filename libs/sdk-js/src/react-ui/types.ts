export interface UIMessage {
  type: "ui";

  id: string;
  name: string;
  props: Record<string, unknown>;
  metadata: {
    run_id?: string;
    name?: string;
    tags?: string[];
    message_id?: string;
    [key: string]: unknown;
  };
}

export interface RemoveUIMessage {
  type: "remove-ui";
  id: string;
}

export function uiMessageReducer(
  state: UIMessage[],
  update: UIMessage | RemoveUIMessage | (UIMessage | RemoveUIMessage)[],
) {
  const events = Array.isArray(update) ? update : [update];
  let newState = state.slice();

  for (const event of events) {
    if (event.type === "remove-ui") {
      newState = newState.filter((ui) => ui.id !== event.id);
      continue;
    }

    const index = state.findIndex((ui) => ui.id === event.id);
    if (index !== -1) {
      newState[index] = event;
    } else {
      newState.push(event);
    }
  }

  return newState;
}
