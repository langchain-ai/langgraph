package com.langgraph.pregel.stream;

import com.langgraph.pregel.StreamMode;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Utility class for formatting output for streaming based on the stream mode.
 */
public class StreamOutput {
    private StreamOutput() {
        // Prevent instantiation
    }
    
    /**
     * Format output for streaming based on the stream mode.
     *
     * @param state Current state
     * @param updatedChannels Set of channel names that were updated in this step
     * @param step Current step number
     * @param hasMoreWork Whether there is more work to do
     * @param streamMode Stream mode
     * @return Formatted output
     */
    public static Map<String, Object> format(
            Map<String, Object> state,
            Set<String> updatedChannels,
            int step,
            boolean hasMoreWork,
            StreamMode streamMode) {
        if (streamMode == null) {
            streamMode = StreamMode.VALUES;
        }
        
        switch (streamMode) {
            case VALUES:
                // Return the full state
                return state != null ? new HashMap<>(state) : new HashMap<>();
                
            case UPDATES:
                // Return only the updated channels
                Map<String, Object> updates = new HashMap<>();
                if (updatedChannels != null && state != null) {
                    for (String channelName : updatedChannels) {
                        if (state.containsKey(channelName)) {
                            updates.put(channelName, state.get(channelName));
                        }
                    }
                }
                return updates;
                
            case DEBUG:
                // Return detailed debug information
                Map<String, Object> debug = new HashMap<>();
                debug.put("state", state != null ? new HashMap<>(state) : new HashMap<>());
                debug.put("updated_channels", updatedChannels);
                debug.put("step", step);
                debug.put("has_more_work", hasMoreWork);
                return debug;
                
            default:
                return state != null ? new HashMap<>(state) : new HashMap<>();
        }
    }
    
    /**
     * Format values mode output.
     *
     * @param state Current state
     * @return Formatted output
     */
    public static Map<String, Object> formatValues(Map<String, Object> state) {
        return format(state, null, 0, false, StreamMode.VALUES);
    }
    
    /**
     * Format updates mode output.
     *
     * @param state Current state
     * @param updatedChannels Set of channel names that were updated in this step
     * @return Formatted output
     */
    public static Map<String, Object> formatUpdates(Map<String, Object> state, Set<String> updatedChannels) {
        return format(state, updatedChannels, 0, false, StreamMode.UPDATES);
    }
    
    /**
     * Format debug mode output.
     *
     * @param state Current state
     * @param updatedChannels Set of channel names that were updated in this step
     * @param step Current step number
     * @param hasMoreWork Whether there is more work to do
     * @return Formatted output
     */
    public static Map<String, Object> formatDebug(
            Map<String, Object> state,
            Set<String> updatedChannels,
            int step,
            boolean hasMoreWork) {
        return format(state, updatedChannels, step, hasMoreWork, StreamMode.DEBUG);
    }
}