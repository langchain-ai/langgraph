package com.langgraph.pregel.task;

import com.langgraph.pregel.PregelNode;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Plans which nodes to execute based on channel updates.
 * 
 * <p>The TaskPlanner determines which nodes to execute in each superstep based on which 
 * channels have been updated. There are two important cases:</p>
 * 
 * <ol>
 *   <li>First Superstep (no channels updated yet): 
 *     <ul>
 *       <li>Current Behavior: All nodes are executed, regardless of subscriptions or triggers</li>
 *       <li>Python LangGraph Behavior: Only nodes with the input channel as their trigger would execute</li>
 *     </ul>
 *   </li>
 *   <li>Subsequent Supersteps:
 *     <ul>
 *       <li>Nodes execute if either:
 *         <ol>
 *           <li>They subscribe to a channel that was updated</li>
 *           <li>They have a trigger matching a channel that was updated</li>
 *         </ol>
 *       </li>
 *     </ul>
 *   </li>
 * </ol>
 * 
 * <p>Note: For Python compatibility, a future version of this implementation will likely change
 * to only execute nodes with the appropriate input channel trigger in the first superstep.</p>
 */
public class TaskPlanner {
    private final Map<String, PregelNode> nodes;
    
    // The input channel name, used to determine which nodes should run in first superstep
    private final String inputChannelName;
    
    /**
     * Create a TaskPlanner with default input channel name "input".
     *
     * @param nodes Map of node names to nodes
     */
    public TaskPlanner(Map<String, PregelNode> nodes) {
        this(nodes, "input");
    }
    
    /**
     * Create a TaskPlanner with a specific input channel name.
     *
     * @param nodes Map of node names to nodes
     * @param inputChannelName The name of the input channel
     */
    public TaskPlanner(Map<String, PregelNode> nodes, String inputChannelName) {
        if (nodes == null) {
            throw new IllegalArgumentException("Nodes cannot be null");
        }
        this.nodes = new HashMap<>(nodes);
        this.inputChannelName = inputChannelName;
    }
    
    /**
     * Plan which nodes to execute based on updated channels.
     * With full Python compatibility for uninitialized channels.
     *
     * @param updatedChannels Set of channel names that were updated
     * @return List of tasks to execute
     */
    public List<PregelTask> plan(Collection<String> updatedChannels) {
        // For first superstep when no channels have been updated yet
        if (updatedChannels == null || updatedChannels.isEmpty()) {
            // Proper Python compatibility: only run nodes with input channel trigger
            List<PregelTask> tasks = new ArrayList<>();
            for (PregelNode node : nodes.values()) {
                // Use the newer method for checking trigger channels
                if (node.isTriggeredBy(inputChannelName)) {
                    // Use the first trigger channel for Task creation
                    String trigger = node.getTriggerChannels().isEmpty() ? 
                        null : node.getTriggerChannels().iterator().next();
                    tasks.add(new PregelTask(node.getName(), trigger, node.getRetryPolicy()));
                }
            }
            return tasks;
        }
        
        // Convert to set for O(1) lookups
        Set<String> updatedChannelSet = new HashSet<>(updatedChannels);
        
        // Collect tasks to execute
        List<PregelTask> tasks = new ArrayList<>();
        
        for (PregelNode node : nodes.values()) {
            // Check if the node reads from any updated channels
            boolean shouldExecute = false;
            
            for (String channelName : node.getChannels()) {
                if (updatedChannelSet.contains(channelName)) {
                    shouldExecute = true;
                    break;
                }
            }
            
            // Check if the node is triggered by any updated channels
            if (!shouldExecute) {
                for (String channelName : node.getTriggerChannels()) {
                    if (updatedChannelSet.contains(channelName)) {
                        shouldExecute = true;
                        break;
                    }
                }
            }
            
            if (shouldExecute) {
                // Use the first trigger channel for Task creation
                String trigger = node.getTriggerChannels().isEmpty() ? 
                    null : node.getTriggerChannels().iterator().next();
                tasks.add(new PregelTask(node.getName(), trigger, node.getRetryPolicy()));
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