package com.langgraph.pregel.state;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Represents a snapshot of execution state at a superstep boundary.
 * Checkpoints contain the serialized state of all channels at a point in time.
 */
public class Checkpoint {
    private Map<String, Object> channelValues;
    
    /**
     * Create a Checkpoint with channel values.
     *
     * @param channelValues Map of channel names to their checkpoint values
     */
    public Checkpoint(Map<String, Object> channelValues) {
        this.channelValues = channelValues != null ? new HashMap<>(channelValues) : new HashMap<>();
    }
    
    /**
     * Create an empty Checkpoint.
     */
    public Checkpoint() {
        this(Collections.emptyMap());
    }
    
    /**
     * Get the channel values.
     *
     * @return Unmodifiable map of channel values
     */
    public Map<String, Object> getValues() {
        return Collections.unmodifiableMap(channelValues);
    }
    
    /**
     * Get a channel value by name.
     *
     * @param channelName Channel name
     * @return Channel value, or null if not present
     */
    public Object getValue(String channelName) {
        return channelValues.get(channelName);
    }
    
    /**
     * Update the checkpoint with new channel values.
     *
     * @param channelValues New channel values
     */
    public void update(Map<String, Object> channelValues) {
        if (channelValues == null) {
            throw new IllegalArgumentException("Channel values cannot be null");
        }
        this.channelValues = new HashMap<>(channelValues);
    }
    
    /**
     * Update a single channel value.
     *
     * @param channelName Channel name
     * @param value Channel value
     */
    public void updateChannel(String channelName, Object value) {
        if (channelName == null || channelName.isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        if (value == null) {
            channelValues.remove(channelName);
        } else {
            channelValues.put(channelName, value);
        }
    }
    
    /**
     * Check if this checkpoint contains a value for the given channel.
     *
     * @param channelName Channel name
     * @return True if the checkpoint contains a value for the channel
     */
    public boolean containsChannel(String channelName) {
        return channelValues.containsKey(channelName);
    }
    
    /**
     * Create a new Checkpoint with updated values.
     *
     * @param updates Map of channel names to values to update
     * @return New Checkpoint with updated values
     */
    public Checkpoint withUpdates(Map<String, Object> updates) {
        if (updates == null || updates.isEmpty()) {
            return this;
        }
        
        Map<String, Object> newValues = new HashMap<>(this.channelValues);
        newValues.putAll(updates);
        return new Checkpoint(newValues);
    }
    
    /**
     * Create a new Checkpoint with only the specified channels.
     *
     * @param channelNames Channel names to include
     * @return New Checkpoint with only the specified channels
     */
    public Checkpoint subset(Iterable<String> channelNames) {
        if (channelNames == null) {
            return new Checkpoint();
        }
        
        Map<String, Object> subsetValues = new HashMap<>();
        for (String name : channelNames) {
            if (channelValues.containsKey(name)) {
                subsetValues.put(name, channelValues.get(name));
            }
        }
        return new Checkpoint(subsetValues);
    }
    
    /**
     * Get the number of channels in this checkpoint.
     *
     * @return Number of channels
     */
    public int size() {
        return channelValues.size();
    }
    
    /**
     * Check if this checkpoint is empty.
     *
     * @return True if the checkpoint contains no values
     */
    public boolean isEmpty() {
        return channelValues.isEmpty();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Checkpoint that = (Checkpoint) o;
        return Objects.equals(channelValues, that.channelValues);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(channelValues);
    }
    
    @Override
    public String toString() {
        return "Checkpoint{channelCount=" + channelValues.size() + "}";
    }
}