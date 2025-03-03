package com.langgraph.channels;

import java.lang.reflect.Type;

/**
 * Abstract base implementation of BaseChannel that provides common functionality.
 *
 * @param <V> Type of the value stored in the channel
 * @param <U> Type of the update received by the channel
 * @param <C> Type of the checkpoint representation
 */
public abstract class AbstractChannel<V, U, C> implements BaseChannel<V, U, C> {
    /**
     * The full generic type information for value type.
     */
    protected final TypeReference<V> valueTypeRef;
    
    /**
     * The full generic type information for update type.
     */
    protected final TypeReference<U> updateTypeRef;
    
    /**
     * The full generic type information for checkpoint type.
     */
    protected final TypeReference<C> checkpointTypeRef;
    
    /**
     * The channel key (name).
     */
    protected String key = "";
    
    /**
     * Creates a new channel with full generic type information.
     * 
     * @param valueTypeRef TypeReference for the value type
     * @param updateTypeRef TypeReference for the update type
     * @param checkpointTypeRef TypeReference for the checkpoint type
     */
    protected AbstractChannel(TypeReference<V> valueTypeRef, TypeReference<U> updateTypeRef, TypeReference<C> checkpointTypeRef) {
        this.valueTypeRef = valueTypeRef;
        this.updateTypeRef = updateTypeRef;
        this.checkpointTypeRef = checkpointTypeRef;
    }
    
    /**
     * Creates a new channel with full generic type information and key.
     * 
     * @param valueTypeRef TypeReference for the value type
     * @param updateTypeRef TypeReference for the update type
     * @param checkpointTypeRef TypeReference for the checkpoint type
     * @param key The key (name) of this channel
     */
    protected AbstractChannel(TypeReference<V> valueTypeRef, TypeReference<U> updateTypeRef, 
                              TypeReference<C> checkpointTypeRef, String key) {
        this.valueTypeRef = valueTypeRef;
        this.updateTypeRef = updateTypeRef;
        this.checkpointTypeRef = checkpointTypeRef;
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
        return valueTypeRef.getRawClass();
    }
    
    @Override
    public Class<U> getUpdateType() {
        return updateTypeRef.getRawClass();
    }
    
    @Override
    public Class<C> getCheckpointType() {
        return checkpointTypeRef.getRawClass();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        
        AbstractChannel<?, ?, ?> that = (AbstractChannel<?, ?, ?>) o;
        
        // Compare key, valueTypeRef, updateTypeRef, and checkpointTypeRef
        // The exact comparison of stored values is responsibility of subclasses
        if (!key.equals(that.key)) return false;
        if (!valueTypeRef.equals(that.valueTypeRef)) return false;
        if (!updateTypeRef.equals(that.updateTypeRef)) return false;
        return checkpointTypeRef.equals(that.checkpointTypeRef);
    }
    
    @Override
    public int hashCode() {
        int result = valueTypeRef.hashCode();
        result = 31 * result + updateTypeRef.hashCode();
        result = 31 * result + checkpointTypeRef.hashCode();
        result = 31 * result + key.hashCode();
        return result;
    }
    
    /**
     * Handle a single-value update when channels support it.
     * This method can be overridden by channels that want to support
     * single-value updates (like TopicChannel). The default implementation
     * returns false, indicating the update was not handled.
     * 
     * @param singleValue The single value to update with
     * @return true if the channel was updated, false otherwise
     */
    public boolean updateSingleValue(U singleValue) {
        // Default implementation doesn't support single value updates
        return false;
    }
}