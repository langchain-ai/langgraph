package com.langgraph.pregel.registry;

import com.langgraph.channels.BaseChannel;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Registry for managing a collection of channels.
 * Provides methods for registration, validation, and channel lookup.
 */
public class ChannelRegistry {
    private final Map<String, BaseChannel> channels;
    
    /**
     * Create an empty ChannelRegistry.
     */
    public ChannelRegistry() {
        this.channels = new HashMap<>();
    }
    
    /**
     * Create a ChannelRegistry with initial channels.
     *
     * @param channels Map of channel names to channels
     */
    public ChannelRegistry(Map<String, BaseChannel> channels) {
        this.channels = new HashMap<>();
        if (channels != null) {
            channels.forEach(this::register);
        }
    }
    
    /**
     * Register a channel.
     *
     * @param name Channel name
     * @param channel Channel to register
     * @return This registry
     * @throws IllegalArgumentException If a channel with the same name is already registered
     */
    public ChannelRegistry register(String name, BaseChannel channel) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        if (channel == null) {
            throw new IllegalArgumentException("Channel cannot be null");
        }
        
        if (channels.containsKey(name)) {
            throw new IllegalArgumentException("Channel with name '" + name + "' is already registered");
        }
        
        channels.put(name, channel);
        return this;
    }
    
    /**
     * Register multiple channels.
     *
     * @param channelsToRegister Map of channel names to channels
     * @return This registry
     * @throws IllegalArgumentException If a channel with the same name is already registered
     */
    public ChannelRegistry registerAll(Map<String, BaseChannel> channelsToRegister) {
        if (channelsToRegister != null) {
            channelsToRegister.forEach(this::register);
        }
        return this;
    }
    
    /**
     * Get a channel by name.
     *
     * @param name Name of the channel to get
     * @return Channel with the given name
     * @throws NoSuchElementException If no channel with the given name is registered
     */
    public BaseChannel get(String name) {
        BaseChannel channel = channels.get(name);
        if (channel == null) {
            throw new NoSuchElementException("No channel registered with name '" + name + "'");
        }
        return channel;
    }
    
    /**
     * Check if a channel with the given name is registered.
     *
     * @param name Name to check
     * @return True if a channel with the given name is registered
     */
    public boolean contains(String name) {
        return channels.containsKey(name);
    }
    
    /**
     * Remove a channel by name.
     *
     * @param name Name of the channel to remove
     * @return This registry
     */
    public ChannelRegistry remove(String name) {
        channels.remove(name);
        return this;
    }
    
    /**
     * Get all registered channels.
     *
     * @return Unmodifiable map of channel names to channels
     */
    public Map<String, BaseChannel> getAll() {
        return Collections.unmodifiableMap(channels);
    }
    
    /**
     * Get the number of registered channels.
     *
     * @return Number of registered channels
     */
    public int size() {
        return channels.size();
    }
    
    /**
     * Get the names of all registered channels.
     *
     * @return Set of channel names
     */
    public Set<String> getNames() {
        return Collections.unmodifiableSet(channels.keySet());
    }
    
    /**
     * Update a channel with a value.
     *
     * @param name Channel name
     * @param value Value to update the channel with
     * @return True if the channel was updated, false otherwise
     * @throws NoSuchElementException If no channel with the given name is registered
     */
    public boolean update(String name, Object value) {
        BaseChannel channel = get(name);
        return channel.update(Collections.singletonList(value));
    }
    
    /**
     * Update multiple channels.
     *
     * @param updates Map of channel names to values
     * @return Set of channel names that were updated
     */
    public Set<String> updateAll(Map<String, Object> updates) {
        if (updates == null || updates.isEmpty()) {
            return Collections.emptySet();
        }
        
        Set<String> updatedChannels = new HashSet<>();
        
        for (Map.Entry<String, Object> entry : updates.entrySet()) {
            String name = entry.getKey();
            Object value = entry.getValue();
            
            if (contains(name) && update(name, value)) {
                updatedChannels.add(name);
            }
        }
        
        return updatedChannels;
    }
    
    /**
     * Collect values from all channels.
     *
     * @return Map of channel names to their current values
     */
    public Map<String, Object> collectValues() {
        Map<String, Object> values = new HashMap<>();
        
        for (Map.Entry<String, BaseChannel> entry : channels.entrySet()) {
            String name = entry.getKey();
            BaseChannel channel = entry.getValue();
            
            Object value = channel.getValue();
            if (value != null) {
                values.put(name, value);
            }
        }
        
        return values;
    }
    
    /**
     * Capture checkpoint data from all channels.
     *
     * @return Map of channel names to their checkpoint data
     */
    public Map<String, Object> checkpoint() {
        Map<String, Object> checkpointData = new HashMap<>();
        
        for (Map.Entry<String, BaseChannel> entry : channels.entrySet()) {
            String name = entry.getKey();
            BaseChannel channel = entry.getValue();
            
            Object data = channel.checkpoint();
            if (data != null) {
                checkpointData.put(name, data);
            }
        }
        
        return checkpointData;
    }
    
    /**
     * Restore channels from checkpoint data.
     *
     * @param checkpointData Map of channel names to checkpoint data
     */
    public void restoreFromCheckpoint(Map<String, Object> checkpointData) {
        if (checkpointData == null || checkpointData.isEmpty()) {
            return;
        }
        
        for (Map.Entry<String, Object> entry : checkpointData.entrySet()) {
            String name = entry.getKey();
            Object data = entry.getValue();
            
            if (contains(name)) {
                channels.get(name).fromCheckpoint(data);
            }
        }
    }
    
    /**
     * Reset all channels, clearing any update flags.
     */
    public void resetUpdated() {
        for (BaseChannel channel : channels.values()) {
            channel.resetUpdated();
        }
    }
    
    /**
     * Get a subset of this registry with channels that match the given names.
     *
     * @param channelNames Names of channels to include
     * @return New registry with only the specified channels
     */
    public ChannelRegistry subset(Collection<String> channelNames) {
        ChannelRegistry subset = new ChannelRegistry();
        
        if (channelNames != null) {
            for (String name : channelNames) {
                if (contains(name)) {
                    subset.register(name, get(name));
                }
            }
        }
        
        return subset;
    }
}