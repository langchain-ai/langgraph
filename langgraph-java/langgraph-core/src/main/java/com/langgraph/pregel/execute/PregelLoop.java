package com.langgraph.pregel.execute;

import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.pregel.StreamMode;
import com.langgraph.pregel.registry.ChannelRegistry;
import com.langgraph.pregel.state.Checkpoint;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Main execution loop for Pregel.
 * Manages the execution of multiple supersteps until completion or interruption.
 */
public class PregelLoop {
    private static final int DEFAULT_MAX_STEPS = 100;
    
    private final SuperstepManager superstepManager;
    private final BaseCheckpointSaver checkpointer;
    private final int maxSteps;
    private final AtomicInteger stepCount;
    
    /**
     * Create a PregelLoop.
     *
     * @param superstepManager Manager for executing supersteps
     * @param checkpointer Optional checkpointer for persisting state
     * @param maxSteps Maximum number of steps to execute before terminating
     */
    public PregelLoop(
            SuperstepManager superstepManager,
            BaseCheckpointSaver checkpointer,
            int maxSteps) {
        this.superstepManager = superstepManager;
        this.checkpointer = checkpointer;
        this.maxSteps = maxSteps > 0 ? maxSteps : DEFAULT_MAX_STEPS;
        this.stepCount = new AtomicInteger(0);
    }
    
    /**
     * Create a PregelLoop with default max steps.
     *
     * @param superstepManager Manager for executing supersteps
     * @param checkpointer Optional checkpointer for persisting state
     */
    public PregelLoop(SuperstepManager superstepManager, BaseCheckpointSaver checkpointer) {
        this(superstepManager, checkpointer, DEFAULT_MAX_STEPS);
    }
    
    /**
     * Create a PregelLoop without checkpointing.
     *
     * @param superstepManager Manager for executing supersteps
     * @param maxSteps Maximum number of steps
     */
    public PregelLoop(SuperstepManager superstepManager, int maxSteps) {
        this(superstepManager, null, maxSteps);
    }
    
    /**
     * Create a PregelLoop with default configuration.
     *
     * @param superstepManager Manager for executing supersteps
     */
    public PregelLoop(SuperstepManager superstepManager) {
        this(superstepManager, null, DEFAULT_MAX_STEPS);
    }
    
    /**
     * Execute the Pregel loop to completion and return the final state.
     *
     * @param input Initial input to the loop
     * @param context Execution context
     * @param threadId Thread ID for checkpointing
     * @return Final state after execution
     */
    public Map<String, Object> execute(
            Map<String, Object> input,
            Map<String, Object> context,
            String threadId) {
        if (input != null && !input.isEmpty()) {
            // Initialize with input
            initializeWithInput(input);
        } else if (threadId != null && checkpointer != null) {
            // Try to restore from checkpoint
            restoreFromCheckpoint(threadId);
        }
        
        // Execute supersteps until completion
        Map<String, Object> result = null;
        stepCount.set(0);
        
        while (stepCount.incrementAndGet() <= maxSteps) {
            // Execute a single superstep
            SuperstepResult stepResult = superstepManager.executeStep(context);
            
            // Capture result
            result = stepResult.getState();
            
            // Create checkpoint if configured
            if (threadId != null && checkpointer != null) {
                createCheckpoint(threadId, result);
            }
            
            // Check if we're done
            if (!stepResult.hasMoreWork()) {
                break;
            }
        }
        
        return result;
    }
    
    /**
     * Execute the Pregel loop with streaming of intermediate states.
     *
     * @param input Initial input to the loop
     * @param context Execution context
     * @param threadId Thread ID for checkpointing
     * @param streamMode Streaming mode
     * @param callback Callback for each state update
     */
    public void stream(
            Map<String, Object> input,
            Map<String, Object> context,
            String threadId,
            StreamMode streamMode,
            Function<Map<String, Object>, Boolean> callback) {
        if (input != null && !input.isEmpty()) {
            // Initialize with input
            initializeWithInput(input);
        } else if (threadId != null && checkpointer != null) {
            // Try to restore from checkpoint
            restoreFromCheckpoint(threadId);
        }
        
        // Execute supersteps until completion
        stepCount.set(0);
        boolean continueExecution = true;
        
        while (continueExecution && stepCount.incrementAndGet() <= maxSteps) {
            // Execute a single superstep
            SuperstepResult stepResult = superstepManager.executeStep(context);
            
            // Stream result based on mode
            Map<String, Object> streamData = formatStreamOutput(stepResult, streamMode);
            
            // Call the callback with the result
            if (callback != null) {
                continueExecution = callback.apply(streamData);
            }
            
            // Create checkpoint if configured
            if (threadId != null && checkpointer != null) {
                createCheckpoint(threadId, stepResult.getState());
            }
            
            // Check if we're done
            if (!stepResult.hasMoreWork()) {
                break;
            }
        }
    }
    
    /**
     * Initialize the Pregel loop with input.
     *
     * @param input Initial input
     */
    private void initializeWithInput(Map<String, Object> input) {
        if (input == null || input.isEmpty()) {
            return;
        }
        
        // Update channel values with input values
        ChannelRegistry channelRegistry = getChannelRegistry();
        boolean anyChannelUpdated = false;
        
        for (Map.Entry<String, Object> entry : input.entrySet()) {
            String channelName = entry.getKey();
            Object value = entry.getValue();
            
            if (channelRegistry.contains(channelName) && value != null) {
                // Update the channel with the input value
                boolean updated = channelRegistry.update(channelName, value);
                if (updated) {
                    anyChannelUpdated = true;
                }
            }
        }
        
        // Mark all input channels as updated for the initial superstep
        superstepManager.addUpdatedChannels(input.keySet());
    }
    
    /**
     * Restore state from checkpoint.
     *
     * @param threadId Thread ID
     * @return True if state was restored, false otherwise
     */
    private boolean restoreFromCheckpoint(String threadId) {
        if (checkpointer == null || threadId == null) {
            return false;
        }
        
        Optional<String> latestCheckpoint = checkpointer.latest(threadId);
        if (!latestCheckpoint.isPresent()) {
            return false;
        }
        
        Optional<Map<String, Object>> checkpoint = checkpointer.getValues(latestCheckpoint.get());
        if (!checkpoint.isPresent()) {
            return false;
        }
        
        // Restore channel values from checkpoint
        ChannelRegistry channelRegistry = getChannelRegistry();
        channelRegistry.restoreFromCheckpoint(checkpoint.get());
        
        // Mark all channels as updated for the first superstep
        superstepManager.addUpdatedChannels(checkpoint.get().keySet());
        
        return true;
    }
    
    /**
     * Create a checkpoint.
     *
     * @param threadId Thread ID
     * @param state Current state
     */
    private void createCheckpoint(String threadId, Map<String, Object> state) {
        if (checkpointer == null || threadId == null) {
            return;
        }
        
        // Create checkpoint with the current state
        // We use threadId as the thread ID and construct a unique checkpoint ID
        checkpointer.checkpoint(threadId, new HashMap<>(state));
    }
    
    /**
     * Format the output for streaming based on the stream mode.
     *
     * @param result Superstep result
     * @param streamMode Stream mode
     * @return Formatted output
     */
    private Map<String, Object> formatStreamOutput(SuperstepResult result, StreamMode streamMode) {
        if (streamMode == null) {
            streamMode = StreamMode.VALUES;
        }
        
        switch (streamMode) {
            case VALUES:
                // Return the full state
                return result.getState();
                
            case UPDATES:
                // Return only the updated channels
                Map<String, Object> updates = new HashMap<>();
                for (String channelName : result.getUpdatedChannels()) {
                    if (result.getState().containsKey(channelName)) {
                        updates.put(channelName, result.getState().get(channelName));
                    }
                }
                return updates;
                
            case DEBUG:
                // Return detailed debug information
                Map<String, Object> debug = new HashMap<>();
                debug.put("state", result.getState());
                debug.put("updated_channels", result.getUpdatedChannels());
                debug.put("step", stepCount.get());
                debug.put("has_more_work", result.hasMoreWork());
                return debug;
                
            default:
                return result.getState();
        }
    }
    
    /**
     * Get the channel registry from the superstep manager.
     *
     * @return Channel registry
     */
    private ChannelRegistry getChannelRegistry() {
        // Access the private channelRegistry field from SuperstepManager for now
        // In an ideal world, SuperstepManager would expose a getter for this
        try {
            java.lang.reflect.Field field = SuperstepManager.class.getDeclaredField("channelRegistry");
            field.setAccessible(true);
            return (ChannelRegistry) field.get(superstepManager);
        } catch (Exception e) {
            throw new RuntimeException("Failed to access channel registry", e);
        }
    }
    
    /**
     * Get the current step count.
     *
     * @return Current step count
     */
    public int getStepCount() {
        return stepCount.get();
    }
    
    /**
     * Reset the step count.
     */
    public void resetStepCount() {
        stepCount.set(0);
    }
}