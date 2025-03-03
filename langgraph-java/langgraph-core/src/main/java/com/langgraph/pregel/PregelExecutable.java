package com.langgraph.pregel;

import java.util.Map;

/**
 * Functional interface for actions that can be executed within Pregel.
 * This represents the computations performed by nodes in the graph with type-safe input and output.
 * 
 * @param <I> The input type that the node expects
 * @param <O> The output type that the node produces
 */
@FunctionalInterface
public interface PregelExecutable<I, O> {
    /**
     * Execute the action with typed inputs from channels and context information.
     *
     * @param inputs Map of channel names to their current values with specified input type
     * @param context Execution context containing thread ID and other configuration
     * @return Map of channel names to values of the specified output type
     */
    Map<String, O> execute(Map<String, I> inputs, Map<String, Object> context);
}