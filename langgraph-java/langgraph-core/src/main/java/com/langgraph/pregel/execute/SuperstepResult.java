package com.langgraph.pregel.execute;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * Represents the result of a single superstep execution.
 * Contains information about channel updates and the current state.
 */
public class SuperstepResult {
    private final boolean hasMoreWork;
    private final Set<String> updatedChannels;
    private final Map<String, Object> state;
    
    /**
     * Create a SuperstepResult.
     *
     * @param hasMoreWork True if there is more work to do (channels were updated)
     * @param updatedChannels Set of channel names that were updated
     * @param state Current state after the superstep
     */
    public SuperstepResult(boolean hasMoreWork, Set<String> updatedChannels, Map<String, Object> state) {
        this.hasMoreWork = hasMoreWork;
        this.updatedChannels = updatedChannels != null 
                ? Collections.unmodifiableSet(updatedChannels) 
                : Collections.emptySet();
        this.state = state != null 
                ? Collections.unmodifiableMap(state) 
                : Collections.emptyMap();
    }
    
    /**
     * Check if there is more work to do.
     *
     * @return True if there is more work to do
     */
    public boolean hasMoreWork() {
        return hasMoreWork;
    }
    
    /**
     * Get the channels that were updated in this superstep.
     *
     * @return Unmodifiable set of updated channel names
     */
    public Set<String> getUpdatedChannels() {
        return updatedChannels;
    }
    
    /**
     * Get the current state after the superstep.
     *
     * @return Unmodifiable map of the current state
     */
    public Map<String, Object> getState() {
        return state;
    }
    
    /**
     * Check if a specific channel was updated.
     *
     * @param channelName Channel name to check
     * @return True if the channel was updated
     */
    public boolean wasChannelUpdated(String channelName) {
        return updatedChannels.contains(channelName);
    }
    
    /**
     * Get the number of updated channels.
     *
     * @return Number of updated channels
     */
    public int getUpdateCount() {
        return updatedChannels.size();
    }
    
    @Override
    public String toString() {
        return "SuperstepResult{" +
                "hasMoreWork=" + hasMoreWork +
                ", updatedChannelCount=" + updatedChannels.size() +
                ", stateSize=" + state.size() +
                '}';
    }
}