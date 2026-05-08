/**
 * Starter LangGraph.js Template
 * Make this code your own!
 */
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateAnnotation } from "./state.js";
import * as crypto from "crypto";

// Approved model registry — only models listed here may be used.
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  // Add approved models here, e.g.:
  // "approved-model-id": "approved-model-version-digest"
};

/**
 * Emit a structured audit record for every AI-driven decision.
 * This record is written to stdout as newline-delimited JSON so it can be
 * captured by any log-aggregation pipeline and treated as an immutable audit trail.
 */
function emitAuditRecord(fields: {
  event: string;
  modelId: string;
  modelVersion: string;
  inputHash: string;
  outputSummary: string;
  timestamp: string;
  principal: string;
}): void {
  // Write to stderr so it is separated from normal application output and
  // can be routed to a dedicated audit sink.
  process.stderr.write(JSON.stringify({ audit: true, ...fields }) + "\n");
}

/**
 * Validate that the requested model is in the approved registry.
 * Throws if the model is not approved.
 */
function assertModelApproved(modelId: string): void {
  if (!Object.prototype.hasOwnProperty.call(APPROVED_MODEL_REGISTRY, modelId)) {
    throw new Error(
      `Model '${modelId}' is not in the organisation's approved model registry. ` +
        `Approved models: ${Object.keys(APPROVED_MODEL_REGISTRY).join(", ") || "(none configured)"}`
    );
  }
}

/**
 * Define a node, these do the work of the graph and should have most of the logic.
 * Must return a subset of the properties set in StateAnnotation.
 * @param state The current state of the graph.
 * @param config Extra parameters passed into the state graph.
 * @returns Some subset of parameters of the graph state, used to update the state
 * for the edges and nodes executed next.
 *
 * NOTE: To integrate a real LLM, add an approved model to APPROVED_MODEL_REGISTRY
 * above and call assertModelApproved(modelId) before invoking it. Do NOT use
 * unapproved models such as Claude (ChatAnthropic) or OpenAI GPT variants.
 * Any model used must be pinned to an immutable version digest and present in
 * the registry.
 */
const callModel = async (
  state: typeof StateAnnotation.State,
  _config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> => {
  const timestamp = new Date().toISOString();
  // Approved model ID must be present in APPROVED_MODEL_REGISTRY.
  const modelId = process.env.APPROVED_MODEL_ID ?? "";
  const modelVersion = process.env.APPROVED_MODEL_VERSION ?? "";

  // Enforce approved model registry check before any LLM interaction.
  assertModelApproved(modelId);

  // Compute a hash of the input for audit purposes (data minimisation: only hash, not raw content).
  const inputHash = crypto
    .createHash("sha256")
    .update(JSON.stringify(state.messages.map((m: { role?: string; _getType?: () => string }) => ({
      role: m.role ?? (m._getType ? m._getType() : "unknown"),
    })))
    .digest("hex");

  // Log the LLM interaction input (minimised: message count and input hash only).
  emitAuditRecord({
    event: "llm_interaction_input",
    modelId,
    modelVersion,
    inputHash,
    outputSummary: "",
    timestamp,
    principal: process.env.AGENT_PRINCIPAL ?? "unknown",
  });

  /**
   * Do some work here using an approved model from APPROVED_MODEL_REGISTRY.
   * Example (replace with an approved model):
   *
   * const approvedModelId = "<approved-model-id-from-registry>";
   * assertModelApproved(approvedModelId);
   * // ... invoke the approved model ...
   */

  // Hardcoded stub response (replace with actual approved-model invocation).
  const responseContent = `Hi there! How are you?`;

  // Provenance metadata for synthetic/AI-generated content.
  const provenanceTag = `[AI-GENERATED | model:${modelId} | version:${modelVersion} | ts:${timestamp}]`;
  const labelledContent = `${responseContent}\n${provenanceTag}`;

  // Compute output hash for audit record.
  const outputHash = crypto
    .createHash("sha256")
    .update(labelledContent)
    .digest("hex");

  // Log the LLM interaction output (minimised: output hash and provenance tag only).
  emitAuditRecord({
    event: "llm_interaction_output",
    modelId,
    modelVersion,
    inputHash,
    outputSummary: outputHash,
    timestamp: new Date().toISOString(),
    principal: process.env.AGENT_PRINCIPAL ?? "unknown",
  });

  return {
    messages: [
      {
        role: "assistant",
        content: labelledContent,
      },
    ],
  };
};

/**
 * Routing function: Determines whether to continue research or end the builder.
 * This function decides if the gathered information is satisfactory or if more research is needed.
 *
 * @param state - The current state of the research builder
 * @returns Either "callModel" to continue research or END to finish the builder
 */
export const route = (
  state: typeof StateAnnotation.State,
): "__end__" | "callModel" => {
  if (state.messages.length > 0) {
    return "__end__";
  }
  // Loop back
  return "callModel";
};

// Finally, create the graph itself.
const builder = new StateGraph(StateAnnotation)
  // Add the nodes to do the work.
  // Chaining the nodes together in this way
  // updates the types of the StateGraph instance
  // so you have static type checking when it comes time
  // to add the edges.
  .addNode("callModel", callModel)
  // Regular edges mean "always transition to node B after node A is done"
  // The "__start__" and "__end__" nodes are "virtual" nodes that are always present
  // and represent the beginning and end of the builder.
  .addEdge("__start__", "callModel")
  // Conditional edges optionally route to different nodes (or end)
  .addConditionalEdges("callModel", route);

export const graph = builder.compile();

graph.name = "New Agent";