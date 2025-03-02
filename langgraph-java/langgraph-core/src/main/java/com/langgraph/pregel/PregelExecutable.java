package com.langgraph.pregel;

import java.util.Map;

/**
 * Functional interface for actions that can be executed within Pregel.
 * This represents the computations performed by nodes in the graph.
 */
@FunctionalInterface
public interface PregelExecutable {
    /**
     * Execute the action with inputs from channels and context information.
     *
     * @param inputs Map of channel names to their current values
     * @param context Execution context containing thread ID and other configuration
     * @return Map of channel names to values to be written/updated
     */
    Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context);
}