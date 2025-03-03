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
     * Creates a new LastValue channel using TypeReference to preserve generic type information.
     * This is especially useful for generic types like List&lt;Integer&gt;.
     *
     * @param typeRef The TypeReference that captures the full generic type
     */
    protected LastValue(TypeReference<V> typeRef) {
        // For LastValue, V=U=C (they are all the same type)
        super(typeRef, typeRef, typeRef);
    }
    
    /**
     * Creates a new LastValue channel using TypeReference to preserve generic type information,
     * with the specified key.
     *
     * @param typeRef The TypeReference that captures the full generic type
     * @param key The key (name) of this channel
     */
    protected LastValue(TypeReference<V> typeRef, String key) {
        // For LastValue, V=U=C (they are all the same type)
        super(typeRef, typeRef, typeRef, key);
    }
    
    /**
     * Factory method to create a LastValue channel with proper generic type capture.
     * Use this instead of constructor when dealing with generic types like List&lt;Integer&gt;.
     * 
     * <p>Example usage:
     * <pre>
     * LastValue&lt;List&lt;Integer&gt;&gt; channel = LastValue.&lt;List&lt;Integer&gt;&gt;create();
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @return A new LastValue channel with the captured type parameter
     */
    public static <T> LastValue<T> create() {
        return new LastValue<>(new TypeReference<T>() {});
    }
    
    /**
     * Factory method to create a LastValue channel with proper generic type capture
     * and a specified key.
     * 
     * <p>Example usage:
     * <pre>
     * LastValue&lt;List&lt;Integer&gt;&gt; channel = LastValue.&lt;List&lt;Integer&gt;&gt;create("myChannel");
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @param key The key (name) for the channel
     * @return A new LastValue channel with the captured type parameter and specified key
     */
    public static <T> LastValue<T> create(String key) {
        return new LastValue<>(new TypeReference<T>() {}, key);
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
        // Return null if not initialized, for Python compatibility
        // This prevents EmptyChannelException when accessing uninitialized channels
        return value;
    }
    
    @Override
    public BaseChannel<V, V, V> fromCheckpoint(V checkpoint) {
        LastValue<V> newChannel = new LastValue<>(valueTypeRef, key);
        
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
        if (!super.equals(obj)) {
            return false;
        }
        
        LastValue<?> other = (LastValue<?>) obj;
        return initialized == other.initialized &&
               (value == null ? other.value == null : value.equals(other.value));
    }
    
    /**
     * Returns the hash code of this channel.
     *
     * @return The hash code
     */
    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (initialized ? 1 : 0);
        result = 31 * result + (value != null ? value.hashCode() : 0);
        return result;
    }
}