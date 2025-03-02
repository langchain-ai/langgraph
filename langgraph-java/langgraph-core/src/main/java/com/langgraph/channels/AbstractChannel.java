package com.langgraph.channels;

/**
 * Abstract base implementation of BaseChannel that provides common functionality.
 *
 * @param <V> Type of the value stored in the channel
 * @param <U> Type of the update received by the channel
 * @param <C> Type of the checkpoint representation
 */
public abstract class AbstractChannel<V, U, C> implements BaseChannel<V, U, C> {
    /**
     * The value type class.
     */
    protected final Class<?> valueType;
    
    /**
     * The channel key (name).
     */
    protected String key = "";
    
    /**
     * Creates a new channel with the specified value type.
     * 
     * @param valueType The class representing the value type of this channel
     */
    protected AbstractChannel(Class<?> valueType) {
        this.valueType = valueType;
    }
    
    /**
     * Creates a new channel with the specified value type and key.
     * 
     * @param valueType The class representing the value type of this channel
     * @param key The key (name) of this channel
     */
    protected AbstractChannel(Class<?> valueType, String key) {
        this.valueType = valueType;
        this.key = key;
    }
    
    @Override
    public String getKey() {
        return key;
    }
    
    @Override
    public void setKey(String key) {
        this.key = key;
    }
    
    /**
     * By default, checkpoint returns the current value.
     * Subclasses can override this if they need different checkpoint behavior.
     */
    @Override
    public C checkpoint() throws EmptyChannelException {
        @SuppressWarnings("unchecked")
        C value = (C) get();
        return value;
    }
    
    /**
     * Returns the value type.
     * 
     * @return The value type
     */
    public Class<?> getValueType() {
        return valueType;
    }
}