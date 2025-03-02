package com.langgraph.channels;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A channel that collects values into a list.
 * Unlike LastValue, it can receive multiple values per update.
 *
 * @param <V> Type of the value stored in the channel
 */
public class TopicChannel<V> extends AbstractChannel<List<V>, V, List<V>> {
    /**
     * The current list of values, empty if the channel has not been updated yet.
     */
    private List<V> values = new ArrayList<>();
    
    /**
     * Flag to track if this channel has been initialized.
     */
    private boolean initialized = false;
    
    /**
     * Flag to determine if the channel should reset after consumption.
     */
    private final boolean resetOnConsume;
    
    /**
     * Creates a new Topic channel with the specified value type.
     * By default, the channel will not reset after consumption.
     *
     * @param valueType The class representing the value type of this channel
     */
    public TopicChannel(Class<V> valueType) {
        this(valueType, false);
    }
    
    /**
     * Creates a new Topic channel with the specified value type and reset behavior.
     *
     * @param valueType The class representing the value type of this channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    public TopicChannel(Class<V> valueType, boolean resetOnConsume) {
        super(valueType);
        this.resetOnConsume = resetOnConsume;
    }
    
    /**
     * Creates a new Topic channel with the specified value type, key, and reset behavior.
     *
     * @param valueType The class representing the value type of this channel
     * @param key The key (name) of this channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    public TopicChannel(Class<V> valueType, String key, boolean resetOnConsume) {
        super(valueType, key);
        this.resetOnConsume = resetOnConsume;
    }
    
    @Override
    public boolean update(List<V> newValues) {
        if (newValues.isEmpty()) {
            return false;
        }
        
        values.addAll(newValues);
        initialized = true;
        return true;
    }
    
    @Override
    public List<V> get() throws EmptyChannelException {
        if (!initialized) {
            throw new EmptyChannelException("Topic channel at key '" + key + "' is empty (never updated)");
        }
        return Collections.unmodifiableList(values);
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public BaseChannel<List<V>, V, List<V>> fromCheckpoint(List<V> checkpoint) {
        TopicChannel<V> newChannel = new TopicChannel<>((Class<V>) valueType, key, resetOnConsume);
        if (checkpoint != null) {
            newChannel.values = new ArrayList<>(checkpoint);
            newChannel.initialized = true;
        }
        return newChannel;
    }
    
    @Override
    public boolean consume() {
        if (resetOnConsume && initialized) {
            values.clear();
            initialized = false;
            return true;
        }
        return false;
    }
    
    /**
     * Returns the string representation of this channel.
     *
     * @return String representation
     */
    @Override
    public String toString() {
        return "Topic(" + (initialized ? values : "empty") + ")";
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
        if (!(obj instanceof TopicChannel)) {
            return false;
        }
        
        TopicChannel<?> other = (TopicChannel<?>) obj;
        return valueType.equals(other.valueType) &&
               key.equals(other.key) &&
               initialized == other.initialized &&
               resetOnConsume == other.resetOnConsume &&
               values.equals(other.values);
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
        result = 31 * result + (resetOnConsume ? 1 : 0);
        result = 31 * result + values.hashCode();
        return result;
    }
}