package com.langgraph.pregel;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Core interface defining the contract for all Pregel implementations.
 * This protocol provides methods for execution, streaming results, and state management.
 */
public interface PregelProtocol {
    /**
     * Invoke the graph with input and run to completion.
     *
     * @param input Input to the graph, typically a map of channel names to values
     * @param config Optional configuration parameters
     * @return Output from the graph after execution completes
     */
    Object invoke(Object input, Map<String, Object> config);
    
    /**
     * Stream execution results as they are produced.
     *
     * @param input Input to the graph, typically a map of channel names to values
     * @param config Optional configuration parameters
     * @param streamMode Mode of streaming (VALUES, UPDATES, or DEBUG)
     * @return Iterator of execution updates
     */
    Iterator<Object> stream(Object input, Map<String, Object> config, StreamMode streamMode);
    
    /**
     * Get the current state for a thread.
     *
     * @param threadId Optional thread ID, if null returns the state for the default thread
     * @return Current state
     */
    Object getState(String threadId);
    
    /**
     * Update the state for a thread.
     *
     * @param threadId Thread ID to update
     * @param state New state to set
     */
    void updateState(String threadId, Object state);
    
    /**
     * Get the state history for a thread.
     *
     * @param threadId Thread ID to get history for
     * @return List of state snapshots in chronological order
     */
    List<Object> getStateHistory(String threadId);
}