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
    protected final Class<V> valueType;
    
    /**
     * The update type class.
     */
    protected final Class<U> updateType;
    
    /**
     * The checkpoint type class.
     */
    protected final Class<C> checkpointType;
    
    /**
     * The channel key (name).
     */
    protected String key = "";
    
    /**
     * Creates a new channel with the specified type information.
     * 
     * @param valueType The class representing the value type of this channel
     * @param updateType The class representing the update type of this channel
     * @param checkpointType The class representing the checkpoint type of this channel
     */
    protected AbstractChannel(Class<V> valueType, Class<U> updateType, Class<C> checkpointType) {
        this.valueType = valueType;
        this.updateType = updateType;
        this.checkpointType = checkpointType;
    }
    
    /**
     * Creates a new channel with the specified type information and key.
     * 
     * @param valueType The class representing the value type of this channel
     * @param updateType The class representing the update type of this channel
     * @param checkpointType The class representing the checkpoint type of this channel
     * @param key The key (name) of this channel
     */
    protected AbstractChannel(Class<V> valueType, Class<U> updateType, Class<C> checkpointType, String key) {
        this.valueType = valueType;
        this.updateType = updateType;
        this.checkpointType = checkpointType;
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
     * Note: This implementation assumes C and V are the same type for most channels.
     * Subclasses where C and V differ MUST override this method.
     */
    @Override
    public C checkpoint() throws EmptyChannelException {
        try {
            // This cast is unavoidable due to Java generics limitations
            // We can't enforce that C = V at compile time, so runtime cast is needed
            // Each subclass properly implements fromCheckpoint to handle this correctly
            @SuppressWarnings("unchecked")
            C value = (C) get();
            return value;
        } catch (EmptyChannelException e) {
            // For Python compatibility, allow checkpointing uninitialized channels
            return null;
        }
    }
    
    @Override
    public Class<V> getValueType() {
        return valueType;
    }
    
    @Override
    public Class<U> getUpdateType() {
        return updateType;
    }
    
    @Override
    public Class<C> getCheckpointType() {
        return checkpointType;
    }
}