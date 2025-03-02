package com.langgraph.pregel;

/**
 * Represents an error that occurs when a graph exceeds its recursion limit
 * during execution.
 */
public class GraphRecursionError extends RuntimeException {
    
    /**
     * Creates a new GraphRecursionError with the specified message.
     *
     * @param message The error message
     */
    public GraphRecursionError(String message) {
        super(message);
    }
    
    /**
     * Creates a new GraphRecursionError with the specified message and cause.
     *
     * @param message The error message
     * @param cause The cause of the error
     */
    public GraphRecursionError(String message, Throwable cause) {
        super(message, cause);
    }
}