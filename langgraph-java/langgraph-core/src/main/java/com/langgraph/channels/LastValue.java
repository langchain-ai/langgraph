package com.langgraph.channels;

import java.util.List;

/**
 * A channel that stores the last value received.
 * Can receive at most one value per update.
 *
 * @param <V> Type of the value stored in the channel
 */
public class LastValue<V> extends AbstractChannel<V, V, V> {
    /**
     * The current value, null if the channel has not been updated yet.
     */
    private V value;
    
    /**
     * Flag to track if this channel has been initialized.
     */
    private boolean initialized = false;
    
    /**
     * Creates a new LastValue channel with the specified value type.
     *
     * @param valueType The class representing the value type of this channel
     */
    public LastValue(Class<V> valueType) {
        super(valueType);
    }
    
    /**
     * Creates a new LastValue channel with the specified value type and key.
     *
     * @param valueType The class representing the value type of this channel
     * @param key The key (name) of this channel
     */
    public LastValue(Class<V> valueType, String key) {
        super(valueType, key);
    }
    
    @Override
    public boolean update(List<V> values) throws InvalidUpdateException {
        if (values.isEmpty()) {
            return false;
        }
        
        if (values.size() > 1) {
            throw new InvalidUpdateException(
                "At key '" + key + "': LastValue channel can receive only one value per update. " +
                "Use a different channel type to handle multiple values.");
        }
        
        value = values.get(0);
        initialized = true;
        return true;
    }
    
    @Override
    public V get() throws EmptyChannelException {
        if (!initialized) {
            throw new EmptyChannelException("LastValue channel at key '" + key + "' is empty (never updated)");
        }
        return value;
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public BaseChannel<V, V, V> fromCheckpoint(V checkpoint) {
        LastValue<V> newChannel = new LastValue<>((Class<V>) valueType, key);
        // Even null is a valid checkpoint value - it means the channel was initialized with null
        newChannel.value = checkpoint;
        newChannel.initialized = true;
        return newChannel;
    }
    
    /**
     * Returns the string representation of this channel.
     *
     * @return String representation
     */
    @Override
    public String toString() {
        return "LastValue(" + (initialized ? value : "empty") + ")";
    }
    
    /**
     * Checks if this channel is equal to another object.
     *
     * @param obj The object to compare with
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof LastValue)) {
            return false;
        }
        
        LastValue<?> other = (LastValue<?>) obj;
        return valueType.equals(other.valueType) &&
               key.equals(other.key) &&
               initialized == other.initialized &&
               (value == null ? other.value == null : value.equals(other.value));
    }
    
    /**
     * Returns the hash code of this channel.
     *
     * @return The hash code
     */
    @Override
    public int hashCode() {
        int result = valueType.hashCode();
        result = 31 * result + key.hashCode();
        result = 31 * result + (initialized ? 1 : 0);
        result = 31 * result + (value != null ? value.hashCode() : 0);
        return result;
    }
}