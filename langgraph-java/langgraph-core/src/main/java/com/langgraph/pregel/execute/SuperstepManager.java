package com.langgraph.pregel.execute;

import com.langgraph.pregel.PregelNode;
import com.langgraph.pregel.registry.ChannelRegistry;
import com.langgraph.pregel.registry.NodeRegistry;
import com.langgraph.pregel.task.PregelExecutableTask;
import com.langgraph.pregel.task.PregelTask;
import com.langgraph.pregel.task.TaskExecutor;
import com.langgraph.pregel.task.TaskPlanner;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

/**
 * Manages the execution of a single superstep in the Pregel system.
 * A superstep consists of planning, execution, and update phases.
 */
public class SuperstepManager {
    private final NodeRegistry nodeRegistry;
    private final ChannelRegistry channelRegistry;
    private final TaskPlanner taskPlanner;
    private final TaskExecutor taskExecutor;
    private final Set<String> updatedChannels;
    
    /**
     * Create a SuperstepManager.
     *
     * @param nodeRegistry Node registry
     * @param channelRegistry Channel registry
     * @param taskPlanner Task planner
     * @param taskExecutor Task executor
     */
    public SuperstepManager(
            NodeRegistry nodeRegistry,
            ChannelRegistry channelRegistry,
            TaskPlanner taskPlanner,
            TaskExecutor taskExecutor) {
        this.nodeRegistry = nodeRegistry;
        this.channelRegistry = channelRegistry;
        this.taskPlanner = taskPlanner;
        this.taskExecutor = taskExecutor;
        this.updatedChannels = new HashSet<>();
    }
    
    /**
     * Create a SuperstepManager with default planner and executor.
     *
     * @param nodeRegistry Node registry
     * @param channelRegistry Channel registry
     */
    public SuperstepManager(NodeRegistry nodeRegistry, ChannelRegistry channelRegistry) {
        this(
                nodeRegistry,
                channelRegistry,
                new TaskPlanner(nodeRegistry.getAll()),
                new TaskExecutor()
        );
    }
    
    /**
     * Execute a single superstep.
     *
     * @param context Execution context
     * @return SuperstepResult containing the result of the superstep
     */
    public SuperstepResult executeStep(Map<String, Object> context) {
        // Plan phase: Determine nodes to execute based on channel updates
        // For Python compatibility, this will now return tasks even if no channels
        // have been updated, ensuring nodes run with uninitialized channels
        List<PregelTask> tasks = taskPlanner.planAndPrioritize(updatedChannels);
        
        if (tasks.isEmpty()) {
            // No tasks to execute, superstep is complete
            return new SuperstepResult(false, Collections.emptySet(), channelRegistry.collectValues());
        }
        
        // Clear updated channels for this superstep
        updatedChannels.clear();
        
        // Execute phase: Run all tasks and collect results
        List<CompletableFuture<Map<String, Object>>> futures = new ArrayList<>();
        Map<PregelTask, CompletableFuture<Map<String, Object>>> taskFutures = new HashMap<>();
        
        for (PregelTask task : tasks) {
            PregelNode node = nodeRegistry.get(task.getNode());
            
            // Prepare inputs for this task
            Map<String, Object> inputs = new HashMap<>();
            for (String channelName : node.getChannels()) {
                if (channelRegistry.contains(channelName)) {
                    inputs.put(channelName, channelRegistry.get(channelName).getValue());
                }
            }
            
            // Add trigger channel values if present
            for (String triggerChannel : node.getTriggerChannels()) {
                if (channelRegistry.contains(triggerChannel)) {
                    inputs.put(triggerChannel, channelRegistry.get(triggerChannel).getValue());
                }
            }
            
            // Create executable task
            PregelExecutableTask executableTask = new PregelExecutableTask(task, inputs, context);
            
            // Execute task asynchronously
            CompletableFuture<Map<String, Object>> future = taskExecutor.executeAsync(node, executableTask);
            futures.add(future);
            taskFutures.put(task, future);
        }
        
        // Wait for all tasks to complete
        CompletableFuture<Void> allFutures = CompletableFuture.allOf(
                futures.toArray(new CompletableFuture[0]));
        
        try {
            // Block until all tasks complete
            allFutures.join();
            
            // Collect results
            Map<String, Set<Object>> allUpdates = new ConcurrentHashMap<>();
            
            for (Map.Entry<PregelTask, CompletableFuture<Map<String, Object>>> entry : taskFutures.entrySet()) {
                PregelTask task = entry.getKey();
                PregelNode node = nodeRegistry.get(task.getNode());
                Map<String, Object> rawResult = entry.getValue().get();
                
                if (rawResult != null) {
                    // Process the output according to write entries
                    Map<String, Object> result = node.processOutput(rawResult);
                    
                    // Record updates for each channel
                    for (Map.Entry<String, Object> update : result.entrySet()) {
                        String channelName = update.getKey();
                        Object value = update.getValue();
                        
                        // Skip null values
                        if (value == null) {
                            continue;
                        }
                        
                        // Group updates by channel name
                        allUpdates.computeIfAbsent(channelName, k -> ConcurrentHashMap.newKeySet())
                                .add(value);
                    }
                }
            }
            
            // Update phase: Apply updates to channels
            Set<String> updated = new HashSet<>();
            
            for (Map.Entry<String, Set<Object>> entry : allUpdates.entrySet()) {
                String channelName = entry.getKey();
                Set<Object> values = entry.getValue();
                
                if (values.size() == 1) {
                    // Single update for this channel
                    Object value = values.iterator().next();
                    if (channelRegistry.update(channelName, value)) {
                        updated.add(channelName);
                    }
                } else if (values.size() > 1) {
                    // Multiple updates for this channel, resolve conflicts
                    // (This could be customized based on channel type)
                    Object lastValue = values.stream().reduce((a, b) -> b).orElse(null);
                    if (lastValue != null && channelRegistry.update(channelName, lastValue)) {
                        updated.add(channelName);
                    }
                }
            }
            
            // Update our tracking of updated channels for the next superstep
            updatedChannels.addAll(updated);
            
            // Return superstep result
            return new SuperstepResult(
                    !updated.isEmpty(),
                    updated,
                    channelRegistry.collectValues()
            );
        } catch (ExecutionException e) {
            // Task execution failed
            Throwable cause = e.getCause();
            throw new SuperstepExecutionException("Superstep execution failed", cause);
        } catch (Exception e) {
            throw new SuperstepExecutionException("Superstep execution failed", e);
        }
    }
    
    /**
     * Get the updated channels from the previous superstep.
     *
     * @return Set of updated channel names
     */
    public Set<String> getUpdatedChannels() {
        return Collections.unmodifiableSet(updatedChannels);
    }
    
    /**
     * Add channels to the set of updated channels.
     *
     * @param channelNames Channel names to add
     */
    public void addUpdatedChannels(Collection<String> channelNames) {
        if (channelNames != null) {
            updatedChannels.addAll(channelNames);
        }
    }
    
    /**
     * Clear the set of updated channels.
     */
    public void clearUpdatedChannels() {
        updatedChannels.clear();
    }
}