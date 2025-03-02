package com.langgraph.pregel.execute;

/**
 * Exception thrown when a superstep execution fails.
 */
public class SuperstepExecutionException extends RuntimeException {
    
    /**
     * Create a SuperstepExecutionException with a message.
     *
     * @param message Error message
     */
    public SuperstepExecutionException(String message) {
        super(message);
    }
    
    /**
     * Create a SuperstepExecutionException with a message and cause.
     *
     * @param message Error message
     * @param cause Cause of the error
     */
    public SuperstepExecutionException(String message, Throwable cause) {
        super(message, cause);
    }
}