import { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import { Annotation, messagesStateReducer } from "@langchain/langgraph";

/**
 * Simple state annotation for the agent
 */
export const StateAnnotation = Annotation.Root({
  /**
   * Messages track the primary execution state of the agent.
   */
  messages: Annotation<BaseMessage[], BaseMessageLike[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});
