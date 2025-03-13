import { bootstrapUiContext } from "./client.js";
bootstrapUiContext();

export { useStreamContext, LoadExternalComponent } from "./client.js";
export {
  uiMessageReducer,
  type UIMessage,
  type RemoveUIMessage,
} from "./types.js";
