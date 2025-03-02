# Java Pregel Core Interfaces

This document defines the Java interfaces for the Pregel execution engine, the computational backbone of LangGraph.

## `PregelProtocol` Interface

The main interface defining the contract for Pregel implementations.

```java
package com.langgraph.pregel;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface defining the contract for all Pregel implementations.
 */
public interface PregelProtocol {
    /**
     * Invoke the graph with input.
     *
     * @param input Input to the graph
     * @param config Optional configuration
     * @return Output from the graph
     */
    Object invoke(Object input, Map<String, Object> config);
    
    /**
     * Stream execution results.
     *
     * @param input Input to the graph
     * @param config Optional configuration
     * @param streamMode Mode of streaming
     * @return Iterator of execution updates
     */
    Iterator<Object> stream(Object input, Map<String, Object> config, StreamMode streamMode);
    
    /**
     * Get the current state.
     *
     * @param threadId Optional thread ID
     * @return Current state
     */
    Object getState(String threadId);
    
    /**
     * Update the state.
     *
     * @param threadId Thread ID
     * @param state New state
     */
    void updateState(String threadId, Object state);
    
    /**
     * Get the state history.
     *
     * @param threadId Thread ID
     * @return List of state snapshots
     */
    List<Object> getStateHistory(String threadId);
}
```

## `StreamMode` Enum

Defines the different streaming options.

```java
package com.langgraph.pregel;

/**
 * Enum defining the different streaming options.
 */
public enum StreamMode {
    /**
     * Stream the complete state after each superstep.
     */
    VALUES,
    
    /**
     * Stream state deltas after each node execution.
     */
    UPDATES,
    
    /**
     * Stream comprehensive execution information for debugging.
     */
    DEBUG
}
```

## `PregelExecutable` Interface

Interface for actions that can be executed within Pregel.

```java
package com.langgraph.pregel;

import java.util.Map;

/**
 * Interface for actions that can be executed within Pregel.
 */
@FunctionalInterface
public interface PregelExecutable {
    /**
     * Execute the action.
     *
     * @param inputs Channel inputs
     * @param context Execution context
     * @return Map of channel updates
     */
    Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context);
}
```

## `PregelNode` Class

Represents an actor in the Pregel system.

```java
package com.langgraph.pregel;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Represents an actor in the Pregel system.
 */
public class PregelNode {
    private final String name;
    private final PregelExecutable action;
    private final Set<String> subscribe;
    private final String trigger;
    private final Set<String> writers;
    private final RetryPolicy retryPolicy;
    
    /**
     * Create a PregelNode.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param subscribe Channel names this node listens to for updates
     * @param trigger Special condition for node execution
     * @param writers Channels this node can write to (for validation)
     * @param retryPolicy Strategy for handling execution failures
     */
    public PregelNode(
            String name,
            PregelExecutable action,
            Collection<String> subscribe,
            String trigger,
            Collection<String> writers,
            RetryPolicy retryPolicy) {
        this.name = name;
        this.action = action;
        this.subscribe = subscribe != null ? new HashSet<>(subscribe) : Collections.emptySet();
        this.trigger = trigger;
        this.writers = writers != null ? new HashSet<>(writers) : Collections.emptySet();
        this.retryPolicy = retryPolicy;
    }
    
    /**
     * Create a PregelNode with default values.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     */
    public PregelNode(String name, PregelExecutable action) {
        this(name, action, null, null, null, null);
    }
    
    /**
     * Get the name of the node.
     *
     * @return Node name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the action to execute.
     *
     * @return Node action
     */
    public PregelExecutable getAction() {
        return action;
    }
    
    /**
     * Get the channels this node subscribes to.
     *
     * @return Set of channel names
     */
    public Set<String> getSubscribe() {
        return Collections.unmodifiableSet(subscribe);
    }
    
    /**
     * Get the trigger condition for this node.
     *
     * @return Trigger condition
     */
    public String getTrigger() {
        return trigger;
    }
    
    /**
     * Get the channels this node can write to.
     *
     * @return Set of channel names
     */
    public Set<String> getWriters() {
        return Collections.unmodifiableSet(writers);
    }
    
    /**
     * Get the retry policy for this node.
     *
     * @return Retry policy
     */
    public RetryPolicy getRetryPolicy() {
        return retryPolicy;
    }
    
    /**
     * Check if this node subscribes to a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node subscribes to the channel
     */
    public boolean subscribesTo(String channelName) {
        return subscribe.contains(channelName);
    }
    
    /**
     * Check if this node has a specific trigger.
     *
     * @param triggerName Trigger name to check
     * @return True if the node has the trigger
     */
    public boolean hasTrigger(String triggerName) {
        return trigger != null && trigger.equals(triggerName);
    }
    
    /**
     * Check if this node can write to a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node can write to the channel
     */
    public boolean canWriteTo(String channelName) {
        return writers.contains(channelName);
    }
}
```

## `PregelTask` and `PregelExecutableTask` Classes

Units of work representing computations to execute.

```java
package com.langgraph.pregel;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Represents a task to be executed within Pregel.
 */
public class PregelTask {
    private final String node;
    private final String trigger;
    private final RetryPolicy retryPolicy;
    
    /**
     * Create a PregelTask.
     *
     * @param node Node name
     * @param trigger Optional trigger
     * @param retryPolicy Optional retry policy
     */
    public PregelTask(String node, String trigger, RetryPolicy retryPolicy) {
        this.node = node;
        this.trigger = trigger;
        this.retryPolicy = retryPolicy;
    }
    
    /**
     * Create a PregelTask with default values.
     *
     * @param node Node name
     */
    public PregelTask(String node) {
        this(node, null, null);
    }
    
    /**
     * Get the node name.
     *
     * @return Node name
     */
    public String getNode() {
        return node;
    }
    
    /**
     * Get the trigger.
     *
     * @return Trigger
     */
    public String getTrigger() {
        return trigger;
    }
    
    /**
     * Get the retry policy.
     *
     * @return Retry policy
     */
    public RetryPolicy getRetryPolicy() {
        return retryPolicy;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PregelTask that = (PregelTask) o;
        return Objects.equals(node, that.node) && 
               Objects.equals(trigger, that.trigger);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(node, trigger);
    }
}

/**
 * Represents an executable task with inputs and context.
 */
public class PregelExecutableTask {
    private final PregelTask task;
    private final Map<String, Object> inputs;
    private final Map<String, Object> context;
    
    /**
     * Create a PregelExecutableTask.
     *
     * @param task Task to execute
     * @param inputs Channel inputs
     * @param context Execution context
     */
    public PregelExecutableTask(
            PregelTask task,
            Map<String, Object> inputs,
            Map<String, Object> context) {
        this.task = task;
        this.inputs = inputs != null ? inputs : Collections.emptyMap();
        this.context = context != null ? context : Collections.emptyMap();
    }
    
    /**
     * Get the task.
     *
     * @return Task
     */
    public PregelTask getTask() {
        return task;
    }
    
    /**
     * Get the inputs.
     *
     * @return Map of channel inputs
     */
    public Map<String, Object> getInputs() {
        return Collections.unmodifiableMap(inputs);
    }
    
    /**
     * Get the context.
     *
     * @return Map of context values
     */
    public Map<String, Object> getContext() {
        return Collections.unmodifiableMap(context);
    }
}
```

## `RetryPolicy` Interface

Interface for handling execution failures.

```java
package com.langgraph.pregel;

/**
 * Interface for handling execution failures.
 */
public interface RetryPolicy {
    /**
     * Decide how to handle a failed execution.
     *
     * @param attempt Current attempt number (1-based)
     * @param error Error that occurred
     * @return Retry decision
     */
    RetryDecision shouldRetry(int attempt, Throwable error);
    
    /**
     * Enum defining retry decisions.
     */
    enum RetryDecision {
        /**
         * Retry the execution
         */
        RETRY,
        
        /**
         * Fail the execution
         */
        FAIL
    }
    
    /**
     * Create a simple retry policy with a maximum number of attempts.
     *
     * @param maxAttempts Maximum number of attempts
     * @return Retry policy
     */
    static RetryPolicy maxAttempts(int maxAttempts) {
        return (attempt, error) -> 
            attempt < maxAttempts ? RetryDecision.RETRY : RetryDecision.FAIL;
    }
    
    /**
     * Create a retry policy that never retries.
     *
     * @return Retry policy
     */
    static RetryPolicy noRetry() {
        return (attempt, error) -> RetryDecision.FAIL;
    }
    
    /**
     * Create a retry policy that always retries.
     *
     * @return Retry policy
     */
    static RetryPolicy alwaysRetry() {
        return (attempt, error) -> RetryDecision.RETRY;
    }
}
```

## `Checkpoint` Class

Represents a snapshot of execution state at a superstep boundary.

```java
package com.langgraph.pregel;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents a snapshot of execution state at a superstep boundary.
 */
public class Checkpoint {
    private Map<String, Object> channelValues;
    
    /**
     * Create a Checkpoint.
     *
     * @param channelValues Channel values
     */
    public Checkpoint(Map<String, Object> channelValues) {
        this.channelValues = new HashMap<>(channelValues);
    }
    
    /**
     * Get the channel values.
     *
     * @return Map of channel values
     */
    public Map<String, Object> getValues() {
        return Collections.unmodifiableMap(channelValues);
    }
    
    /**
     * Update the channel values.
     *
     * @param channelValues New channel values
     */
    public void update(Map<String, Object> channelValues) {
        this.channelValues = new HashMap<>(channelValues);
    }
}
```

## `Pregel` Class (Core Implementation)

The central implementation of the Pregel execution engine.

```java
package com.langgraph.pregel;

import com.langgraph.channels.Channel;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Implementation of the Pregel execution engine.
 */
public class Pregel implements PregelProtocol {
    private final Map<String, PregelNode> nodes;
    private final Map<String, Channel> channels;
    private final BaseCheckpointSaver checkpointer;
    private final ExecutorService executor;
    
    /**
     * Create a Pregel instance.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     * @param checkpointer Optional checkpointer
     */
    public Pregel(
            Map<String, PregelNode> nodes,
            Map<String, Channel> channels,
            BaseCheckpointSaver checkpointer) {
        this.nodes = new HashMap<>(nodes);
        this.channels = new HashMap<>(channels);
        this.checkpointer = checkpointer;
        this.executor = Executors.newWorkStealingPool();
    }
    
    @Override
    public Object invoke(Object input, Map<String, Object> config) {
        // Initialize execution context
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Initialize or restore channel values
        initializeChannels(threadId, input);
        
        // Execute to completion
        List<Object> result = new ArrayList<>();
        for (Object update : executeToCompletion(threadId, context)) {
            result.add(update);
        }
        
        // Return final state
        return result.isEmpty() ? null : result.get(result.size() - 1);
    }
    
    @Override
    public Iterator<Object> stream(Object input, Map<String, Object> config, StreamMode streamMode) {
        // Initialize execution context
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Initialize or restore channel values
        initializeChannels(threadId, input);
        
        // Execute and stream results
        return executeToCompletion(threadId, context);
    }
    
    @Override
    public Object getState(String threadId) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        return captureState();
    }
    
    @Override
    public void updateState(String threadId, Object state) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (!(state instanceof Map)) {
            throw new IllegalArgumentException("State must be a Map");
        }
        
        @SuppressWarnings("unchecked")
        Map<String, Object> stateMap = (Map<String, Object>) state;
        
        // Update channels with the state
        for (Map.Entry<String, Object> entry : stateMap.entrySet()) {
            String channelName = entry.getKey();
            Object value = entry.getValue();
            
            if (channels.containsKey(channelName)) {
                channels.get(channelName).update(value);
            }
        }
        
        // Create a checkpoint
        if (checkpointer != null) {
            checkpointer.checkpoint(threadId, captureChannelValues());
        }
    }
    
    @Override
    public List<Object> getStateHistory(String threadId) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (checkpointer == null) {
            return Collections.emptyList();
        }
        
        List<String> checkpoints = checkpointer.list(threadId);
        List<Object> history = new ArrayList<>();
        
        for (String checkpointId : checkpoints) {
            Optional<Map<String, Object>> values = checkpointer.getValues(checkpointId);
            values.ifPresent(history::add);
        }
        
        return history;
    }
    
    /**
     * Get the thread ID from the configuration.
     *
     * @param config Configuration
     * @return Thread ID
     */
    private String getThreadId(Map<String, Object> config) {
        if (config == null || !config.containsKey("thread_id")) {
            return UUID.randomUUID().toString();
        }
        
        return config.get("thread_id").toString();
    }
    
    /**
     * Create the execution context.
     *
     * @param threadId Thread ID
     * @param config Configuration
     * @return Context map
     */
    private Map<String, Object> createContext(String threadId, Map<String, Object> config) {
        Map<String, Object> context = new HashMap<>();
        context.put("thread_id", threadId);
        
        if (config != null) {
            context.putAll(config);
        }
        
        return context;
    }
    
    /**
     * Initialize or restore channel values.
     *
     * @param threadId Thread ID
     * @param input Input to the graph
     */
    private void initializeChannels(String threadId, Object input) {
        // Check for existing checkpoint
        if (checkpointer != null) {
            Optional<String> latestCheckpoint = checkpointer.latest(threadId);
            
            if (latestCheckpoint.isPresent()) {
                // Restore from checkpoint
                Optional<Map<String, Object>> values = checkpointer.getValues(latestCheckpoint.get());
                
                if (values.isPresent()) {
                    for (Map.Entry<String, Object> entry : values.get().entrySet()) {
                        String channelName = entry.getKey();
                        Object value = entry.getValue();
                        
                        if (channels.containsKey(channelName)) {
                            channels.get(channelName).fromCheckpoint(value);
                        }
                    }
                    
                    return;
                }
            }
        }
        
        // Initialize with input
        if (input instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> inputMap = (Map<String, Object>) input;
            
            for (Map.Entry<String, Object> entry : inputMap.entrySet()) {
                String channelName = entry.getKey();
                Object value = entry.getValue();
                
                if (channels.containsKey(channelName)) {
                    channels.get(channelName).update(value);
                }
            }
        }
    }
    
    /**
     * Execute the graph to completion.
     *
     * @param threadId Thread ID
     * @param context Execution context
     * @return Iterator of execution updates
     */
    private Iterator<Object> executeToCompletion(String threadId, Map<String, Object> context) {
        return new Iterator<Object>() {
            private boolean hasMore = true;
            private final Set<String> updatedChannels = new HashSet<>();
            
            @Override
            public boolean hasNext() {
                return hasMore;
            }
            
            @Override
            public Object next() {
                if (!hasMore) {
                    throw new NoSuchElementException();
                }
                
                // Identify active nodes
                List<PregelTask> tasks = planSuperstep(updatedChannels);
                
                if (tasks.isEmpty()) {
                    hasMore = false;
                    return captureState();
                }
                
                // Reset updated channels for this superstep
                updatedChannels.clear();
                
                // Execute all tasks
                executeSuperstep(tasks, context, updatedChannels);
                
                // Create checkpoint if needed
                if (checkpointer != null) {
                    checkpointer.checkpoint(threadId, captureChannelValues());
                }
                
                // Capture current state
                Object state = captureState();
                
                // Check if we're done
                hasMore = !updatedChannels.isEmpty();
                
                return state;
            }
        };
    }
    
    /**
     * Plan which nodes to execute in the current superstep.
     *
     * @param updatedChannels Set of channel names that were updated
     * @return List of tasks to execute
     */
    private List<PregelTask> planSuperstep(Set<String> updatedChannels) {
        List<PregelTask> tasks = new ArrayList<>();
        
        for (PregelNode node : nodes.values()) {
            // Check if the node subscribes to any updated channels
            boolean shouldExecute = false;
            
            for (String channelName : node.getSubscribe()) {
                if (updatedChannels.contains(channelName)) {
                    shouldExecute = true;
                    break;
                }
            }
            
            // Check if the node has a trigger
            if (node.getTrigger() != null && updatedChannels.contains(node.getTrigger())) {
                shouldExecute = true;
            }
            
            if (shouldExecute) {
                tasks.add(new PregelTask(node.getName(), node.getTrigger(), node.getRetryPolicy()));
            }
        }
        
        return tasks;
    }
    
    /**
     * Execute all tasks in the current superstep.
     *
     * @param tasks Tasks to execute
     * @param context Execution context
     * @param updatedChannels Set to track which channels were updated
     */
    private void executeSuperstep(
            List<PregelTask> tasks,
            Map<String, Object> context,
            Set<String> updatedChannels) {
        // Create executable tasks
        List<PregelExecutableTask> executableTasks = new ArrayList<>();
        
        for (PregelTask task : tasks) {
            // Get inputs for the node
            Map<String, Object> inputs = new HashMap<>();
            PregelNode node = nodes.get(task.getNode());
            
            for (String channelName : node.getSubscribe()) {
                if (channels.containsKey(channelName)) {
                    inputs.put(channelName, channels.get(channelName).getValue());
                }
            }
            
            // Add trigger value if present
            if (task.getTrigger() != null && channels.containsKey(task.getTrigger())) {
                inputs.put(task.getTrigger(), channels.get(task.getTrigger()).getValue());
            }
            
            executableTasks.add(new PregelExecutableTask(task, inputs, context));
        }
        
        // Execute tasks in parallel
        List<Map<String, Object>> results = executeTasks(executableTasks);
        
        // Apply updates
        for (Map<String, Object> updates : results) {
            if (updates != null) {
                for (Map.Entry<String, Object> entry : updates.entrySet()) {
                    String channelName = entry.getKey();
                    Object value = entry.getValue();
                    
                    if (channels.containsKey(channelName)) {
                        boolean wasUpdated = channels.get(channelName).update(value);
                        if (wasUpdated) {
                            updatedChannels.add(channelName);
                        }
                    }
                }
            }
        }
        
        // Reset updated flags on channels
        for (Channel channel : channels.values()) {
            if (channel instanceof Resettable) {
                ((Resettable) channel).resetUpdated();
            }
        }
    }
    
    /**
     * Execute tasks in parallel.
     *
     * @param tasks Tasks to execute
     * @return List of results
     */
    private List<Map<String, Object>> executeTasks(List<PregelExecutableTask> tasks) {
        List<CompletableFuture<Map<String, Object>>> futures = new ArrayList<>();
        
        for (PregelExecutableTask task : tasks) {
            CompletableFuture<Map<String, Object>> future = CompletableFuture.supplyAsync(() -> {
                try {
                    PregelNode node = nodes.get(task.getTask().getNode());
                    return node.getAction().execute(task.getInputs(), task.getContext());
                } catch (Exception e) {
                    // Handle retry logic
                    RetryPolicy retryPolicy = task.getTask().getRetryPolicy();
                    if (retryPolicy != null) {
                        // Retry logic would be implemented here
                        // For simplicity, we're just letting it fail
                    }
                    
                    throw new RuntimeException(
                            "Error executing node: " + task.getTask().getNode(), e);
                }
            }, executor);
            
            futures.add(future);
        }
        
        // Wait for all tasks to complete
        try {
            return futures.stream()
                    .map(CompletableFuture::join)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            // Handle task execution failures
            throw new RuntimeException("Error executing tasks", e);
        }
    }
    
    /**
     * Capture the current channel values.
     *
     * @return Map of channel values
     */
    private Map<String, Object> captureChannelValues() {
        Map<String, Object> values = new HashMap<>();
        
        for (Map.Entry<String, Channel> entry : channels.entrySet()) {
            String channelName = entry.getKey();
            Channel channel = entry.getValue();
            
            Object value = channel.checkpoint();
            if (value != null) {
                values.put(channelName, value);
            }
        }
        
        return values;
    }
    
    /**
     * Capture the current state.
     *
     * @return State map
     */
    private Map<String, Object> captureState() {
        Map<String, Object> state = new HashMap<>();
        
        for (Map.Entry<String, Channel> entry : channels.entrySet()) {
            String channelName = entry.getKey();
            Channel channel = entry.getValue();
            
            Object value = channel.getValue();
            if (value != null) {
                state.put(channelName, value);
            }
        }
        
        return state;
    }
    
    /**
     * Interface for channels that can be reset.
     */
    private interface Resettable {
        void resetUpdated();
    }
    
    /**
     * Builder for creating Pregel instances.
     */
    public static class Builder {
        private final Map<String, PregelNode> nodes = new HashMap<>();
        private final Map<String, Channel> channels = new HashMap<>();
        private BaseCheckpointSaver checkpointer;
        
        /**
         * Add a node.
         *
         * @param node Node to add
         * @return This builder
         */
        public Builder addNode(PregelNode node) {
            nodes.put(node.getName(), node);
            return this;
        }
        
        /**
         * Add a channel.
         *
         * @param name Channel name
         * @param channel Channel to add
         * @return This builder
         */
        public Builder addChannel(String name, Channel channel) {
            channels.put(name, channel);
            return this;
        }
        
        /**
         * Set the checkpointer.
         *
         * @param checkpointer Checkpointer to use
         * @return This builder
         */
        public Builder setCheckpointer(BaseCheckpointSaver checkpointer) {
            this.checkpointer = checkpointer;
            return this;
        }
        
        /**
         * Build the Pregel instance.
         *
         * @return Pregel instance
         */
        public Pregel build() {
            return new Pregel(nodes, channels, checkpointer);
        }
    }
}
```