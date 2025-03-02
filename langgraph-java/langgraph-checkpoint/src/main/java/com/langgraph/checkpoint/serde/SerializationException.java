package com.langgraph.checkpoint.serde;

/**
 * Exception thrown during serialization/deserialization.
 */
public class SerializationException extends RuntimeException {
    /**
     * Create a new serialization exception with a message.
     *
     * @param message Error message
     */
    public SerializationException(String message) {
        super(message);
    }
    
    /**
     * Create a new serialization exception with a message and cause.
     *
     * @param message Error message
     * @param cause Underlying cause
     */
    public SerializationException(String message, Throwable cause) {
        super(message, cause);
    }
}