import { bootstrapUiContext } from "./client.js";
bootstrapUiContext();

export {
  useStreamContext,
  LoadExternalComponent,
  experimental_loadShare,
} from "./client.js";
export {
  uiMessageReducer,
  type UIMessage,
  type RemoveUIMessage,
} from "./types.js";
