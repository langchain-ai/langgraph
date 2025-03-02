package com.langgraph.pregel;

/**
 * Enum defining the different streaming options for Pregel execution.
 */
public enum StreamMode {
    /**
     * Stream the complete state after each superstep.
     */
    VALUES,
    
    /**
     * Stream state deltas after each node execution.
     */
    UPDATES,
    
    /**
     * Stream comprehensive execution information for debugging.
     */
    DEBUG
}