package com.langgraph.pregel.task;

import com.langgraph.pregel.PregelNode;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Plans which nodes to execute based on channel updates.
 */
public class TaskPlanner {
    private final Map<String, PregelNode> nodes;
    
    /**
     * Create a TaskPlanner.
     *
     * @param nodes Map of node names to nodes
     */
    public TaskPlanner(Map<String, PregelNode> nodes) {
        if (nodes == null) {
            throw new IllegalArgumentException("Nodes cannot be null");
        }
        this.nodes = new HashMap<>(nodes);
    }
    
    /**
     * Plan which nodes to execute based on updated channels.
     *
     * @param updatedChannels Set of channel names that were updated
     * @return List of tasks to execute
     */
    public List<PregelTask> plan(Collection<String> updatedChannels) {
        if (updatedChannels == null || updatedChannels.isEmpty()) {
            return Collections.emptyList();
        }
        
        // Convert to set for O(1) lookups
        Set<String> updatedChannelSet = new HashSet<>(updatedChannels);
        
        // Collect tasks to execute
        List<PregelTask> tasks = new ArrayList<>();
        
        for (PregelNode node : nodes.values()) {
            // Check if the node subscribes to any updated channels
            boolean shouldExecute = false;
            
            for (String channelName : node.getSubscribe()) {
                if (updatedChannelSet.contains(channelName)) {
                    shouldExecute = true;
                    break;
                }
            }
            
            // Check if the node has a trigger
            if (!shouldExecute && node.getTrigger() != null && updatedChannelSet.contains(node.getTrigger())) {
                shouldExecute = true;
            }
            
            if (shouldExecute) {
                tasks.add(new PregelTask(node.getName(), node.getTrigger(), node.getRetryPolicy()));
            }
        }
        
        return tasks;
    }
    
    /**
     * Prioritize tasks for execution.
     * This method can be overridden to implement custom prioritization logic.
     *
     * @param tasks List of tasks to prioritize
     * @return Prioritized list of tasks
     */
    public List<PregelTask> prioritize(List<PregelTask> tasks) {
        // Default implementation does not change the order
        return new ArrayList<>(tasks);
    }
    
    /**
     * Filter tasks based on dependencies.
     * This method can be overridden to implement custom filtering logic.
     *
     * @param tasks List of tasks to filter
     * @return Filtered list of tasks
     */
    protected List<PregelTask> filter(List<PregelTask> tasks) {
        return tasks.stream()
                .filter(task -> nodes.containsKey(task.getNode()))
                .collect(Collectors.toList());
    }
    
    /**
     * Plan, filter, and prioritize tasks for execution.
     *
     * @param updatedChannels Set of channel names that were updated
     * @return Prioritized list of tasks to execute
     */
    public List<PregelTask> planAndPrioritize(Collection<String> updatedChannels) {
        List<PregelTask> tasks = plan(updatedChannels);
        tasks = filter(tasks);
        return prioritize(tasks);
    }
}