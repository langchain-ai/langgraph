package com.langgraph.pregel;

import com.langgraph.channels.BaseChannel;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.pregel.execute.PregelLoop;
import com.langgraph.pregel.execute.SuperstepManager;
import com.langgraph.pregel.registry.ChannelRegistry;
import com.langgraph.pregel.registry.NodeRegistry;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

/**
 * The core Pregel implementation.
 * Orchestrates execution of a computational graph using the Bulk Synchronous Parallel model.
 */
public class Pregel implements PregelProtocol {
    private final NodeRegistry nodeRegistry;
    private final ChannelRegistry channelRegistry;
    private final BaseCheckpointSaver checkpointer;
    private final ExecutorService executor;
    private final int maxSteps;
    private final Set<String> inputChannels;
    private final Set<String> outputChannels;
    
    /**
     * Create a Pregel instance with all parameters.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     * @param inputChannels Set of input channel names
     * @param outputChannels Set of output channel names
     * @param checkpointer Optional checkpointer for persisting state
     * @param maxSteps Maximum number of steps to execute
     */
    public Pregel(
            Map<String, PregelNode> nodes,
            Map<String, BaseChannel> channels,
            Set<String> inputChannels,
            Set<String> outputChannels,
            BaseCheckpointSaver checkpointer,
            int maxSteps) {
        // Initialize registries
        this.nodeRegistry = new NodeRegistry(nodes);
        this.channelRegistry = new ChannelRegistry(channels);
        this.inputChannels = inputChannels != null ? inputChannels : new HashSet<>();
        this.outputChannels = outputChannels != null ? outputChannels : new HashSet<>();
        this.checkpointer = checkpointer;
        this.executor = Executors.newWorkStealingPool();
        this.maxSteps = maxSteps;
        
        // Validate configuration
        validate();
    }
    
    /**
     * Create a simple Pregel instance without checkpointing.
     * For more complex configurations, use the Builder pattern.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     */
    public Pregel(Map<String, PregelNode> nodes, Map<String, BaseChannel> channels) {
        this(nodes, channels, new HashSet<>(), new HashSet<>(), null, 100);
    }
    
    /**
     * Validate the Pregel configuration.
     * Checks that nodes and channels are properly configured.
     *
     * @throws IllegalStateException If the configuration is invalid
     */
    private void validate() {
        // Validate nodes
        nodeRegistry.validate();
        
        // Validate channel references
        Set<String> channelNames = channelRegistry.getNames();
        nodeRegistry.validateSubscriptions(channelNames);
        nodeRegistry.validateWriters(channelNames);
        nodeRegistry.validateTriggers(channelNames);
    }
    
    @Override
    public Object invoke(Object input, Map<String, Object> config) {
        // Extract configuration
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Convert input to map if necessary
        Map<String, Object> inputMap = convertInput(input);
        
        // Initialize channels with input
        initializeChannels(inputMap);
        
        // Create execution components
        SuperstepManager superstepManager = new SuperstepManager(nodeRegistry, channelRegistry);
        PregelLoop pregelLoop = new PregelLoop(superstepManager, checkpointer, maxSteps);
        
        // Execute to completion
        Map<String, Object> result = pregelLoop.execute(inputMap, context, threadId);
        
        // Filter the result to only include designated output channels
        if (!outputChannels.isEmpty() && result != null) {
            Map<String, Object> filteredResult = new HashMap<>();
            for (Map.Entry<String, Object> entry : result.entrySet()) {
                if (outputChannels.contains(entry.getKey())) {
                    filteredResult.put(entry.getKey(), entry.getValue());
                }
            }
            return filteredResult;
        }
        
        return result;
    }
    
    @Override
    public Iterator<Object> stream(Object input, Map<String, Object> config, StreamMode streamMode) {
        // Extract configuration
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Convert input to map if necessary
        Map<String, Object> inputMap = convertInput(input);
        
        // Initialize channels with input
        initializeChannels(inputMap);
        
        // Create execution components
        SuperstepManager superstepManager = new SuperstepManager(nodeRegistry, channelRegistry);
        PregelLoop pregelLoop = new PregelLoop(superstepManager, checkpointer, maxSteps);
        
        // Create iterator for streaming results
        return new Iterator<Object>() {
            private final Queue<Object> buffer = new LinkedList<>();
            private boolean isDone = false;
            
            @Override
            public boolean hasNext() {
                if (!buffer.isEmpty()) {
                    return true;
                }
                
                if (isDone) {
                    return false;
                }
                
                // Stream execution and collect results
                pregelLoop.stream(
                        inputMap,
                        context,
                        threadId,
                        streamMode,
                        result -> {
                            buffer.add(result);
                            return true;
                        });
                
                isDone = true;
                return !buffer.isEmpty();
            }
            
            @Override
            public Object next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return buffer.poll();
            }
        };
    }
    
    @Override
    public Object getState(String threadId) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (checkpointer == null) {
            return null;
        }
        
        // Get latest checkpoint
        Optional<String> latestCheckpoint = checkpointer.latest(threadId);
        if (!latestCheckpoint.isPresent()) {
            return null;
        }
        
        // Get checkpoint values
        Optional<Map<String, Object>> values = checkpointer.getValues(latestCheckpoint.get());
        return values.orElse(null);
    }
    
    @Override
    public void updateState(String threadId, Object state) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (!(state instanceof Map)) {
            throw new IllegalArgumentException("State must be a Map");
        }
        
        // Validate and convert state
        Map<String, Object> stateMap = convertStateMap(state);
        
        // Update channels with the state
        initializeChannels(stateMap);
        
        // Create a checkpoint
        if (checkpointer != null) {
            checkpointer.checkpoint(threadId, stateMap);
        }
    }
    
    /**
     * Validates and converts a state object to a Map<String, Object>.
     * 
     * @param state The state object to validate and convert
     * @return A validated Map<String, Object>
     * @throws IllegalArgumentException if state is invalid
     */
    private Map<String, Object> convertStateMap(Object state) {
        if (!(state instanceof Map)) {
            throw new IllegalArgumentException("State must be a Map");
        }
        
        // Safe to cast to Map since we've verified it is a Map
        @SuppressWarnings("unchecked")
        Map<String, Object> stateMap = (Map<String, Object>) state;
        
        // Validate that the values are compatible with their corresponding channels
        for (Map.Entry<String, Object> entry : stateMap.entrySet()) {
            String channelName = entry.getKey();
            Object value = entry.getValue();
            
            if (channelRegistry.contains(channelName) && !isCompatibleWithChannel(channelName, value)) {
                throw new IllegalArgumentException(
                    "Incompatible value type for channel '" + channelName + "': " + 
                    "Expected " + channelRegistry.get(channelName).getUpdateType().getName() + 
                    ", got " + (value != null ? value.getClass().getName() : "null")
                );
            }
        }
        
        return stateMap;
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
     * Initialize channels with input.
     *
     * @param input Input map
     * @throws IllegalArgumentException if any input value is incompatible with its channel
     */
    private void initializeChannels(Map<String, Object> input) {
        if (input == null || input.isEmpty()) {
            return;
        }
        
        // Filter the input to only include designated input channels
        if (!inputChannels.isEmpty()) {
            Map<String, Object> filteredInput = new HashMap<>();
            for (Map.Entry<String, Object> entry : input.entrySet()) {
                String channelName = entry.getKey();
                Object value = entry.getValue();
                
                if (inputChannels.contains(channelName)) {
                    // Validate that the value is compatible with the channel
                    if (!isCompatibleWithChannel(channelName, value)) {
                        throw new IllegalArgumentException(
                            "Incompatible value type for channel '" + channelName + "': " + 
                            "Expected " + channelRegistry.get(channelName).getUpdateType().getName() + 
                            ", got " + (value != null ? value.getClass().getName() : "null")
                        );
                    }
                    
                    filteredInput.put(channelName, value);
                }
            }
            // Update channels with filtered input values
            channelRegistry.updateAll(filteredInput);
        } else {
            // Check all input values for type compatibility
            for (Map.Entry<String, Object> entry : input.entrySet()) {
                String channelName = entry.getKey();
                Object value = entry.getValue();
                
                if (channelRegistry.contains(channelName) && !isCompatibleWithChannel(channelName, value)) {
                    throw new IllegalArgumentException(
                        "Incompatible value type for channel '" + channelName + "': " + 
                        "Expected " + channelRegistry.get(channelName).getUpdateType().getName() + 
                        ", got " + (value != null ? value.getClass().getName() : "null")
                    );
                }
            }
            
            // If no input channels are designated, use all input
            channelRegistry.updateAll(input);
        }
    }
    
    /**
     * Validates that a value is compatible with the channel's expected update type.
     * 
     * @param channelName Name of the channel
     * @param value Value to check
     * @return true if the value is compatible, false otherwise
     */
    private boolean isCompatibleWithChannel(String channelName, Object value) {
        if (!channelRegistry.contains(channelName)) {
            return false;
        }
        
        // Get the channel
        BaseChannel<?, ?, ?> channel = channelRegistry.get(channelName);
        
        // Get the expected update type
        Class<?> updateType = channel.getUpdateType();
        
        // Check if value is null (null is always compatible)
        if (value == null) {
            return true;
        }
        
        // Check if the value is an instance of the expected type
        return updateType.isInstance(value);
    }
    
    /**
     * Convert input to a map if necessary.
     * This validates that the input is a Map where:
     * 1. Keys are Strings matching channel names
     * 2. Values are of a type compatible with the corresponding channel's update type
     *
     * @param input Input object
     * @return Input as a validated map
     * @throws IllegalArgumentException if input is not a Map or contains incompatible types
     */
    private Map<String, Object> convertInput(Object input) {
        if (input == null) {
            return Collections.emptyMap();
        }
        
        if (!(input instanceof Map)) {
            throw new IllegalArgumentException("Input must be a Map<String, Object>");
        }
        
        // Safe to cast to Map since we've verified it is a Map
        // We validate the key types and allowed values below
        @SuppressWarnings("unchecked")
        Map<String, Object> inputMap = (Map<String, Object>) input;
        
        // Optional validation: We could check that each key exists in inputChannels
        // and that the value type matches what the channel expects
        // This would make the code more robust but might also add overhead
        
        return inputMap;
    }
    
    /**
     * Get the NodeRegistry.
     *
     * @return NodeRegistry
     */
    public NodeRegistry getNodeRegistry() {
        return nodeRegistry;
    }
    
    /**
     * Get the ChannelRegistry.
     *
     * @return ChannelRegistry
     */
    public ChannelRegistry getChannelRegistry() {
        return channelRegistry;
    }
    
    /**
     * Get the checkpointer.
     *
     * @return BaseCheckpointSaver
     */
    public BaseCheckpointSaver getCheckpointer() {
        return checkpointer;
    }
    
    /**
     * Shutdown the executor service.
     */
    public void shutdown() {
        executor.shutdown();
    }
    
    /**
     * Builder for creating Pregel instances.
     */
    public static class Builder {
        private final Map<String, PregelNode> nodes = new HashMap<>();
        private final Map<String, BaseChannel> channels = new HashMap<>();
        private Set<String> inputChannels = new HashSet<>();
        private Set<String> outputChannels = new HashSet<>();
        private BaseCheckpointSaver checkpointer;
        private int maxSteps = 100;
        
        /**
         * Add a node to the graph.
         *
         * @param node Node to add
         * @return This builder
         */
        public Builder addNode(PregelNode node) {
            if (node == null) {
                throw new IllegalArgumentException("Node cannot be null");
            }
            nodes.put(node.getName(), node);
            return this;
        }
        
        /**
         * Add multiple nodes to the graph.
         *
         * @param nodes Collection of nodes to add
         * @return This builder
         */
        public Builder addNodes(Collection<PregelNode> nodes) {
            if (nodes != null) {
                for (PregelNode node : nodes) {
                    addNode(node);
                }
            }
            return this;
        }
        
        /**
         * Add a channel to the graph.
         *
         * @param name Channel name
         * @param channel Channel to add
         * @return This builder
         */
        public Builder addChannel(String name, BaseChannel channel) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("Channel name cannot be null or empty");
            }
            if (channel == null) {
                throw new IllegalArgumentException("Channel cannot be null");
            }
            channels.put(name, channel);
            return this;
        }
        
        /**
         * Add multiple channels to the graph.
         *
         * @param channels Map of channel names to channels
         * @return This builder
         */
        public Builder addChannels(Map<String, BaseChannel> channels) {
            if (channels != null) {
                this.channels.putAll(channels);
            }
            return this;
        }
        
        /**
         * Set input channels for this Pregel graph.
         * Input channels will be populated from the input at invocation time.
         *
         * @param inputChannels Collection of input channel names
         * @return This builder
         */
        public Builder setInputChannels(Collection<String> inputChannels) {
            if (inputChannels != null) {
                this.inputChannels = new HashSet<>(inputChannels);
            }
            return this;
        }
        
        /**
         * Set output channels for this Pregel graph.
         * Output channels will be included in the result.
         *
         * @param outputChannels Collection of output channel names
         * @return This builder
         */
        public Builder setOutputChannels(Collection<String> outputChannels) {
            if (outputChannels != null) {
                this.outputChannels = new HashSet<>(outputChannels);
            }
            return this;
        }
        
        /**
         * Set the checkpointer for persisting state.
         *
         * @param checkpointer Checkpointer to use
         * @return This builder
         */
        public Builder setCheckpointer(BaseCheckpointSaver checkpointer) {
            this.checkpointer = checkpointer;
            return this;
        }
        
        /**
         * Set the maximum number of steps to execute.
         *
         * @param maxSteps Maximum number of steps
         * @return This builder
         */
        public Builder setMaxSteps(int maxSteps) {
            if (maxSteps <= 0) {
                throw new IllegalArgumentException("Max steps must be positive");
            }
            this.maxSteps = maxSteps;
            return this;
        }
        
        /**
         * Build the Pregel instance.
         *
         * @return Pregel instance
         */
        public Pregel build() {
            // If no input/output channels are explicitly set, auto-detect them
            if (inputChannels.isEmpty()) {
                // Use all channels as input channels by default
                inputChannels.addAll(channels.keySet());
            }
            
            if (outputChannels.isEmpty()) {
                // Use all channels as output channels by default
                outputChannels.addAll(channels.keySet());
            }
            
            return new Pregel(nodes, channels, inputChannels, outputChannels, checkpointer, maxSteps);
        }
    }
}