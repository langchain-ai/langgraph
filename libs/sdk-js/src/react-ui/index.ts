import { bootstrapUiContext } from "./client.js";
bootstrapUiContext();

export {
  useStreamContext,
  LoadExternalComponent,
  experimental_loadShare,
} from "./client.js";
export {
  uiMessageReducer,
  isUIMessage,
  isRemoveUIMessage,
  type UIMessage,
  type RemoveUIMessage,
} from "./types.js";
