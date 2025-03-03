package com.langgraph.pregel;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Core interface defining the contract for all type-safe Pregel implementations.
 * This protocol provides methods for execution, streaming results, and state management
 * with proper generic type parameters.
 *
 * @param <I> The input type for the graph
 * @param <O> The output type for the graph
 */
public interface PregelProtocol<I, O> {
    /**
     * Invoke the graph with typed input and run to completion.
     *
     * @param input Input to the graph, as a map of channel names to typed values
     * @param config Optional configuration parameters
     * @return Output from the graph after execution completes, as a map of channel names to typed values
     */
    Map<String, O> invoke(Map<String, I> input, Map<String, Object> config);
    
    /**
     * Stream execution results as they are produced, with proper type safety.
     *
     * @param input Input to the graph, as a map of channel names to typed values
     * @param config Optional configuration parameters
     * @param streamMode Mode of streaming (VALUES, UPDATES, or DEBUG)
     * @return Iterator of execution updates with proper types
     */
    Iterator<Map<String, O>> stream(Map<String, I> input, Map<String, Object> config, StreamMode streamMode);
    
    /**
     * Get the current state for a thread with proper type safety.
     *
     * @param threadId Thread ID to get state for
     * @return Current state as a map of channel names to typed values
     */
    Map<String, O> getState(String threadId);
    
    /**
     * Update the state for a thread with type-safe values.
     *
     * @param threadId Thread ID to update
     * @param state New state to set, as a map of channel names to typed values
     */
    void updateState(String threadId, Map<String, O> state);
    
    /**
     * Get the state history for a thread with proper type safety.
     *
     * @param threadId Thread ID to get history for
     * @return List of state snapshots in chronological order, each as a map of channel names to typed values
     */
    List<Map<String, O>> getStateHistory(String threadId);
}