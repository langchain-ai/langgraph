package com.langgraph.channels;

import java.util.List;

/**
 * Base interface for all channels in LangGraph.
 * Channels are the primary mechanism for passing data between nodes in a LangGraph
 * computational graph. They implement different semantics for handling updates
 * (e.g., storing just the last value, aggregating values, etc.)
 *
 * @param <V> Type of the value stored in the channel
 * @param <U> Type of the update received by the channel
 * @param <C> Type of the checkpoint representation
 */
public interface BaseChannel<V, U, C> {
    /**
     * Returns the current value of the channel without type safety checks.
     * This is mainly used internally by the framework.
     *
     * @return Current value as Object
     * @throws EmptyChannelException if the channel has not been updated yet
     */
    default Object getValue() throws EmptyChannelException {
        return get();
    }
    
    /**
     * Marks the channel as updated. This is mainly used internally.
     */
    default void resetUpdated() {
        // Default implementation does nothing
    }
    /**
     * Updates the channel with a sequence of values.
     * The order of the updates in the list is arbitrary.
     * 
     * @param values List of update values
     * @return true if the channel was updated, false otherwise
     * @throws InvalidUpdateException if the update is invalid for this channel type
     */
    boolean update(List<U> values) throws InvalidUpdateException;
    
    /**
     * Returns the current value of the channel.
     * 
     * @return Current value
     * @throws EmptyChannelException if the channel has not been updated yet
     */
    V get() throws EmptyChannelException;
    
    /**
     * Creates a checkpoint of the channel's current state.
     * 
     * @return A serializable representation of the channel's state
     * @throws EmptyChannelException if the channel has not been updated yet
     */
    C checkpoint() throws EmptyChannelException;
    
    /**
     * Creates a new channel instance from a checkpoint.
     * 
     * @param checkpoint Checkpoint data, or null if no prior state
     * @return A new channel instance with the state from the checkpoint
     */
    BaseChannel<V, U, C> fromCheckpoint(C checkpoint);
    
    /**
     * Marks the current value as consumed.
     * By default, this is a no-op.
     * 
     * @return true if the channel was updated, false otherwise
     */
    default boolean consume() {
        return false;
    }
    
    /**
     * Returns the key/name of this channel.
     * 
     * @return Channel key/name
     */
    String getKey();
    
    /**
     * Sets the key/name for this channel.
     * 
     * @param key Channel key/name
     */
    void setKey(String key);
}