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
     * The element type class.
     */
    private final Class<V> elementType;
    
    /**
     * Creates a new Topic channel with the specified value type.
     * By default, the channel will not reset after consumption.
     *
     * @param elementType The class representing the element type within the list
     */
    public TopicChannel(Class<V> elementType) {
        this(elementType, false);
    }
    
    /**
     * Creates a new Topic channel with the specified value type and reset behavior.
     *
     * @param elementType The class representing the element type within the list
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    @SuppressWarnings("unchecked")
    public TopicChannel(Class<V> elementType, boolean resetOnConsume) {
        // For TopicChannel: 
        // - Value type is List<V> but at runtime we can only get List.class
        // - Update type is V (single elements are added)
        // - Checkpoint type is List<V> (same as value type)
        super(
            (Class<List<V>>) (Class<?>) List.class, // Value type (List<V>)
            elementType,                            // Update type (V)
            (Class<List<V>>) (Class<?>) List.class  // Checkpoint type (List<V>)
        );
        this.elementType = elementType;
        this.resetOnConsume = resetOnConsume;
    }
    
    /**
     * Creates a new Topic channel with the specified value type, key, and reset behavior.
     *
     * @param elementType The class representing the element type within the list
     * @param key The key (name) of this channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    @SuppressWarnings("unchecked")
    public TopicChannel(Class<V> elementType, String key, boolean resetOnConsume) {
        // For TopicChannel: 
        // - Value type is List<V> but at runtime we can only get List.class
        // - Update type is V (single elements are added)
        // - Checkpoint type is List<V> (same as value type)
        super(
            (Class<List<V>>) (Class<?>) List.class, // Value type (List<V>)
            elementType,                            // Update type (V)
            (Class<List<V>>) (Class<?>) List.class, // Checkpoint type (List<V>)
            key
        );
        this.elementType = elementType;
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
        // Always return the current list (empty or not) for Python compatibility
        // This prevents EmptyChannelException when accessing uninitialized channels
        return Collections.unmodifiableList(values);
    }
    
    @Override
    public BaseChannel<List<V>, V, List<V>> fromCheckpoint(List<V> checkpoint) {
        TopicChannel<V> newChannel = new TopicChannel<>(elementType, key, resetOnConsume);
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
     * Returns the element type class.
     *
     * @return The element type class
     */
    public Class<V> getElementType() {
        return elementType;
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
        return elementType.equals(other.elementType) &&
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
        int result = elementType.hashCode();
        result = 31 * result + key.hashCode();
        result = 31 * result + (initialized ? 1 : 0);
        result = 31 * result + (resetOnConsume ? 1 : 0);
        result = 31 * result + values.hashCode();
        return result;
    }
}