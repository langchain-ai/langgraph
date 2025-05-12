export interface UIMessage<
  TName extends string = string,
  TProps extends Record<string, unknown> = Record<string, unknown>,
> {
  type: "ui";

  id: string;
  name: TName;
  props: TProps;
  metadata: {
    merge?: boolean;
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

export function isUIMessage(message: unknown): message is UIMessage {
  if (typeof message !== "object" || message == null) return false;
  if (!("type" in message)) return false;
  return message.type === "ui";
}

export function isRemoveUIMessage(
  message: unknown,
): message is RemoveUIMessage {
  if (typeof message !== "object" || message == null) return false;
  if (!("type" in message)) return false;
  return message.type === "remove-ui";
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
      newState[index] = event.metadata.merge
        ? { ...event, props: { ...state[index].props, ...event.props } }
        : event;
    } else {
      newState.push(event);
    }
  }

  return newState;
}
