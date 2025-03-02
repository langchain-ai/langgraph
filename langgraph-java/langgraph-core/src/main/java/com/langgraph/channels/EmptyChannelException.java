package com.langgraph.channels;

/**
 * Exception thrown when trying to access a value from a channel that hasn't been
 * updated yet.
 */
public class EmptyChannelException extends RuntimeException {
    /**
     * Creates a new EmptyChannelException.
     */
    public EmptyChannelException() {
        super("Channel is empty (never updated)");
    }
    
    /**
     * Creates a new EmptyChannelException with a custom message.
     * 
     * @param message The error message
     */
    public EmptyChannelException(String message) {
        super(message);
    }
}