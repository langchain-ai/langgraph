package com.langgraph.channels;

/**
 * Exception thrown when an invalid update is attempted on a channel.
 */
public class InvalidUpdateException extends RuntimeException {
    /**
     * Creates a new InvalidUpdateException.
     */
    public InvalidUpdateException() {
        super("Invalid update for channel");
    }
    
    /**
     * Creates a new InvalidUpdateException with a custom message.
     * 
     * @param message The error message
     */
    public InvalidUpdateException(String message) {
        super(message);
    }
}