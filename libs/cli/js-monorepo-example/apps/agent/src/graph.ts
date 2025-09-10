/**
 * Simple LangGraph.js example for monorepo testing
 */
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateAnnotation } from "./state.js";
import { getGreeting } from "@js-monorepo-example/shared";

/**
 * Simple node that uses the shared library
 */
const callModel = async (
  state: typeof StateAnnotation.State,
  _config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> => {
  // Use functions from the shared library
  const greeting = getGreeting();

  return {
    messages: [
      {
        role: "assistant",
        content: `${greeting}`,
      },
    ],
  };
};

/**
 * Simple routing function
 */
export const route = (
  state: typeof StateAnnotation.State,
): "__end__" | "callModel" => {
  if (state.messages.length > 0) {
    return "__end__";
  }
  return "callModel";
};

// Create the graph
const builder = new StateGraph(StateAnnotation)
  .addNode("callModel", callModel)
  .addEdge("__start__", "callModel")
  .addConditionalEdges("callModel", route);

export const graph = builder.compile();
