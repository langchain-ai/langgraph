/**
 * Simple LangGraph.js example for monorepo testing
 */
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateAnnotation } from "./state.js";
import { getGreeting } from "@js-monorepo-example/shared";
import * as crypto from "crypto";

/**
 * Simple node that uses the shared library
 */
const callModel = async (
  state: typeof StateAnnotation.State,
  _config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> => {
  const interactionTimestamp = new Date().toISOString();
  const modelIdentifier = "js-monorepo-example-agent";
  const modelVersion = "1.0.0";

  // Log interaction start (audit trail)
  const inputHash = crypto
    .createHash("sha256")
    .update(JSON.stringify(state.messages))
    .digest("hex");

  console.log(
    JSON.stringify({
      event: "llm_interaction_start",
      timestamp: interactionTimestamp,
      model: modelIdentifier,
      modelVersion,
      inputHash,
      messageCount: state.messages.length,
    }),
  );

  // Use functions from the shared library
  const greeting = getGreeting();

  const responseContent = `${greeting}`;

  const outputHash = crypto
    .createHash("sha256")
    .update(responseContent)
    .digest("hex");

  // Log interaction completion with decision audit record
  console.log(
    JSON.stringify({
      event: "llm_interaction_complete",
      timestamp: new Date().toISOString(),
      model: modelIdentifier,
      modelVersion,
      inputHash,
      outputHash,
      syntheticContent: true,
      contentOrigin: "ai-generated",
    }),
  );

  return {
    messages: [
      {
        role: "assistant",
        // Provenance label: AI-generated content with model identifier and timestamp
        content: `${responseContent}`,
        // Attach provenance metadata as additional properties
        ...({
          "x-ai-generated": true,
          "x-model-id": modelIdentifier,
          "x-model-version": modelVersion,
          "x-generation-timestamp": interactionTimestamp,
          "x-content-origin": "ai-generated",
          "x-output-hash": outputHash,
        } as object),
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