package com.langgraph.pregel.task;

/**
 * Exception thrown when task execution fails after all retry attempts.
 */
public class TaskExecutionException extends RuntimeException {
    private final int attempt;
    
    /**
     * Create a TaskExecutionException with a message.
     *
     * @param message Error message
     */
    public TaskExecutionException(String message) {
        super(message);
        this.attempt = 0;
    }
    
    /**
     * Create a TaskExecutionException with a message and cause.
     *
     * @param message Error message
     * @param cause Cause of the error
     */
    public TaskExecutionException(String message, Throwable cause) {
        super(message, cause);
        this.attempt = 0;
    }
    
    /**
     * Create a TaskExecutionException with a message, cause, and attempt number.
     *
     * @param message Error message
     * @param cause Cause of the error
     * @param attempt Attempt number that failed
     */
    public TaskExecutionException(String message, Throwable cause, int attempt) {
        super(message, cause);
        this.attempt = attempt;
    }
    
    /**
     * Get the attempt number that failed.
     *
     * @return Attempt number
     */
    public int getAttempt() {
        return attempt;
    }
}