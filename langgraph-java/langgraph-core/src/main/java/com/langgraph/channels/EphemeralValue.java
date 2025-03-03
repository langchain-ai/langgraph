package com.langgraph.channels;

import java.util.List;

/**
 * A channel that stores the last value received but doesn't persist it across checkpoints.
 * This is useful for values that should not be saved in the persistent state.
 *
 * @param <V> Type of the value stored in the channel
 */
public class EphemeralValue<V> extends AbstractChannel<V, V, Void> {
    /**
     * The current value, null if the channel has not been updated yet.
     */
    private V value;
    
    /**
     * Flag to track if this channel has been initialized.
     */
    private boolean initialized = false;
    
    /**
     * Creates a new EphemeralValue channel using TypeReference with the specified value type.
     *
     * @param valueTypeRef TypeReference capturing the value type
     */
    @SuppressWarnings("unchecked")
    protected EphemeralValue(TypeReference<V> valueTypeRef) {
        // For EphemeralValue, V=U but C is Void (always null in checkpoint)
        super(valueTypeRef, valueTypeRef, new TypeReference<Void>() {});
    }
    
    /**
     * Creates a new EphemeralValue channel using TypeReference with specified key.
     *
     * @param valueTypeRef TypeReference capturing the value type
     * @param key The key (name) of this channel
     */
    @SuppressWarnings("unchecked")
    protected EphemeralValue(TypeReference<V> valueTypeRef, String key) {
        // For EphemeralValue, V=U but C is Void (always null in checkpoint)
        super(valueTypeRef, valueTypeRef, new TypeReference<Void>() {}, key);
    }
    
    /**
     * Factory method to create an EphemeralValue channel with proper generic type capture.
     * 
     * <p>Example usage:
     * <pre>
     * EphemeralValue&lt;String&gt; channel = EphemeralValue.&lt;String&gt;create();
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @return A new EphemeralValue channel with the captured type parameter
     */
    public static <T> EphemeralValue<T> create() {
        return new EphemeralValue<>(new TypeReference<T>() {});
    }
    
    /**
     * Factory method to create an EphemeralValue channel with proper generic type capture
     * and a specified key.
     * 
     * <p>Example usage:
     * <pre>
     * EphemeralValue&lt;String&gt; channel = EphemeralValue.&lt;String&gt;create("myChannel");
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @param key The key (name) for the channel
     * @return A new EphemeralValue channel with the captured type parameter and specified key
     */
    public static <T> EphemeralValue<T> create(String key) {
        return new EphemeralValue<>(new TypeReference<T>() {}, key);
    }
    
    @Override
    public boolean update(List<V> values) throws InvalidUpdateException {
        if (values.isEmpty()) {
            return false;
        }
        
        if (values.size() > 1) {
            throw new InvalidUpdateException(
                "At key '" + key + "': EphemeralValue channel can receive only one value per update. " +
                "Use a different channel type to handle multiple values.");
        }
        
        value = values.get(0);
        initialized = true;
        return true;
    }
    
    @Override
    public V get() throws EmptyChannelException {
        if (!initialized) {
            throw new EmptyChannelException("EphemeralValue channel at key '" + key + "' is empty (never updated)");
        }
        return value;
    }
    
    @Override
    public Void checkpoint() {
        // Ephemeral values don't persist in checkpoints
        return null;
    }
    
    @Override
    public BaseChannel<V, V, Void> fromCheckpoint(Void checkpoint) {
        // Always start from an empty state, regardless of checkpoint
        return new EphemeralValue<>(valueTypeRef, key);
    }
    
    /**
     * Returns the string representation of this channel.
     *
     * @return String representation
     */
    @Override
    public String toString() {
        return "EphemeralValue(" + (initialized ? value : "empty") + ")";
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
        if (!(obj instanceof EphemeralValue)) {
            return false;
        }
        if (!super.equals(obj)) {
            return false;
        }
        
        EphemeralValue<?> other = (EphemeralValue<?>) obj;
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