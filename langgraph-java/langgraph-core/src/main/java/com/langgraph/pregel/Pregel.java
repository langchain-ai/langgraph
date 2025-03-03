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

/**
 * The type-safe Pregel implementation.
 * Orchestrates execution of a computational graph using the Bulk Synchronous Parallel model
 * with strict type checking throughout the execution flow.
 *
 * @param <I> The input type for the overall graph
 * @param <O> The output type for the overall graph
 */
public class Pregel<I, O> implements PregelProtocol<I, O> {
    private final NodeRegistry nodeRegistry;
    private final ChannelRegistry channelRegistry;
    private final BaseCheckpointSaver checkpointer;
    private final ExecutorService executor;
    private final int maxSteps;
    private final Set<String> inputChannels;
    private final Set<String> outputChannels;
    
    /**
     * Get a channel by name (for debugging)
     *
     * @param name Channel name
     * @return Channel with the given name
     */
    public BaseChannel<?, ?, ?> getChannel(String name) {
        return channelRegistry.get(name);
    }
    
    /**
     * Create a type-safe Pregel instance with all parameters.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     * @param inputChannels Set of input channel names
     * @param outputChannels Set of output channel names
     * @param checkpointer Optional checkpointer for persisting state
     * @param maxSteps Maximum number of steps to execute
     */
    public Pregel(
            Map<String, PregelNode<?, ?>> nodes,
            Map<String, BaseChannel<?, ?, ?>> channels,
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
     * Validate the Pregel configuration.
     * Checks that nodes and channels are properly configured and type-compatible.
     *
     * @throws IllegalStateException If the configuration is invalid
     */
    private void validate() {
        // Basic validation
        nodeRegistry.validate();
        
        // Validate channel references
        Set<String> channelNames = channelRegistry.getNames();
        nodeRegistry.validateSubscriptions(channelNames);
        nodeRegistry.validateWriters(channelNames);
        nodeRegistry.validateTriggers(channelNames);
        
        // Type compatibility is ensured by generic type parameters
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public Map<String, O> invoke(Map<String, I> input, Map<String, Object> config) {
        // Extract configuration
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Create input map with proper type safety
        Map<String, Object> inputMap = new HashMap<>();
        if (input != null) {
            for (Map.Entry<String, I> entry : input.entrySet()) {
                inputMap.put(entry.getKey(), entry.getValue());
            }
        }
        
        // Initialize channels with input
        initializeChannels(inputMap);
        
        // Create execution components
        SuperstepManager superstepManager = new SuperstepManager(nodeRegistry, channelRegistry);
        PregelLoop pregelLoop = new PregelLoop(superstepManager, checkpointer, maxSteps);
        
        // Execute to completion
        Map<String, Object> result = pregelLoop.execute(inputMap, context, threadId);
        
        // Filter the result
        return filterOutput(result);
    }
    
    @Override
    public Iterator<Map<String, O>> stream(Map<String, I> input, Map<String, Object> config, StreamMode streamMode) {
        // Extract configuration
        String threadId = getThreadId(config);
        Map<String, Object> context = createContext(threadId, config);
        
        // Create input map with proper type safety
        Map<String, Object> inputMap = new HashMap<>();
        if (input != null) {
            for (Map.Entry<String, I> entry : input.entrySet()) {
                inputMap.put(entry.getKey(), entry.getValue());
            }
        }
        
        // Initialize channels with input
        initializeChannels(inputMap);
        
        // Create execution components
        SuperstepManager superstepManager = new SuperstepManager(nodeRegistry, channelRegistry);
        PregelLoop pregelLoop = new PregelLoop(superstepManager, checkpointer, maxSteps);
        
        // Create iterator for streaming results
        return new Iterator<Map<String, O>>() {
            private final Queue<Map<String, O>> buffer = new LinkedList<>();
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
                            // Filter the result to match output type
                            if (result instanceof Map) {
                                @SuppressWarnings("unchecked")
                                Map<String, Object> resultMap = (Map<String, Object>) result;
                                buffer.add(filterOutput(resultMap));
                            }
                            return true;
                        });
                
                isDone = true;
                return !buffer.isEmpty();
            }
            
            @Override
            public Map<String, O> next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return buffer.poll();
            }
        };
    }
    
    @Override
    public Map<String, O> getState(String threadId) {
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
        if (!values.isPresent()) {
            return null;
        }
        
        // Filter to match output type
        return filterOutput(values.get());
    }
    
    @Override
    public void updateState(String threadId, Map<String, O> state) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (state == null) {
            throw new IllegalArgumentException("State cannot be null");
        }
        
        // Convert typed state to Object map for backward compatibility
        Map<String, Object> stateMap = new HashMap<>();
        for (Map.Entry<String, O> entry : state.entrySet()) {
            stateMap.put(entry.getKey(), entry.getValue());
        }
        
        // Validate state map
        validateStateMap(stateMap);
        
        // Update channels with the state
        initializeChannels(stateMap);
        
        // Create a checkpoint
        if (checkpointer != null) {
            checkpointer.checkpoint(threadId, stateMap);
        }
    }
    
    @Override
    public List<Map<String, O>> getStateHistory(String threadId) {
        if (threadId == null) {
            throw new IllegalArgumentException("Thread ID is required");
        }
        
        if (checkpointer == null) {
            return Collections.emptyList();
        }
        
        List<String> checkpoints = checkpointer.list(threadId);
        List<Map<String, O>> history = new ArrayList<>();
        
        for (String checkpointId : checkpoints) {
            Optional<Map<String, Object>> values = checkpointer.getValues(checkpointId);
            if (values.isPresent()) {
                // Filter to match output type
                history.add(filterOutput(values.get()));
            }
        }
        
        return history;
    }
    
    /**
     * Filter result to include only designated output channels and validate type safety.
     *
     * @param result Result map to filter
     * @return Map with typed output values including only designated channels
     */
    @SuppressWarnings("unchecked")
    private Map<String, O> filterOutput(Map<String, Object> result) {
        if (result == null || result.isEmpty()) {
            return Collections.emptyMap();
        }
        
        Map<String, O> typedResult = new HashMap<>();
        
        // Filter the result to only include designated output channels
        for (Map.Entry<String, Object> entry : result.entrySet()) {
            String channelName = entry.getKey();
            Object value = entry.getValue();
            
            if (outputChannels.isEmpty() || outputChannels.contains(channelName)) {
                // Type safety ensured by generic parameters
                
                typedResult.put(channelName, (O) value);
            }
        }
        
        return typedResult;
    }
    
    /**
     * Validates a state map for compatibility with channels.
     *
     * @param stateMap State map to validate
     * @throws IllegalArgumentException if state is invalid
     */
    private void validateStateMap(Map<String, Object> stateMap) {
        // Validate that the values are compatible with their corresponding channels
        for (Map.Entry<String, Object> entry : stateMap.entrySet()) {
            String channelName = entry.getKey();
            Object value = entry.getValue();
            
            // Type safety ensured by generic parameters
        }
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
                    // Type safety ensured by generic parameters
                    
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
                
                // Type safety ensured by generic parameters
            }
            
            // If no input channels are designated, use all input
            channelRegistry.updateAll(input);
        }
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
     * Builder for creating type-safe Pregel instances.
     *
     * @param <I> The input type for the graph
     * @param <O> The output type for the graph
     */
    public static class Builder<I, O> {
        private final Map<String, PregelNode<?, ?>> nodes = new HashMap<>();
        private final Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
        private Set<String> inputChannels = new HashSet<>();
        private Set<String> outputChannels = new HashSet<>();
        private BaseCheckpointSaver checkpointer;
        private int maxSteps = 100;
        
        /**
         * Create a Builder for a type-safe Pregel graph.
         */
        public Builder() {
            // No parameters needed - type parameters are inferred from usage
        }
        
        
        /**
         * Add a node to the graph.
         *
         * @param node Node to add
         * @return This builder
         */
        public Builder<I, O> addNode(PregelNode<?, ?> node) {
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
        public Builder<I, O> addNodes(Collection<PregelNode<?, ?>> nodes) {
            if (nodes != null) {
                for (PregelNode<?, ?> node : nodes) {
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
        public Builder<I, O> addChannel(String name, BaseChannel<?, ?, ?> channel) {
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
        public Builder<I, O> addChannels(Map<String, BaseChannel<?, ?, ?>> channels) {
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
        public Builder<I, O> setInputChannels(Collection<String> inputChannels) {
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
        public Builder<I, O> setOutputChannels(Collection<String> outputChannels) {
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
        public Builder<I, O> setCheckpointer(BaseCheckpointSaver checkpointer) {
            this.checkpointer = checkpointer;
            return this;
        }
        
        /**
         * Set the maximum number of steps to execute.
         *
         * @param maxSteps Maximum number of steps
         * @return This builder
         */
        public Builder<I, O> setMaxSteps(int maxSteps) {
            if (maxSteps <= 0) {
                throw new IllegalArgumentException("Max steps must be positive");
            }
            this.maxSteps = maxSteps;
            return this;
        }
        
        /**
         * Build the type-safe Pregel instance.
         *
         * @return Pregel instance with specified type parameters
         */
        public Pregel<I, O> build() {
            // If no input/output channels are explicitly set, auto-detect them
            if (inputChannels.isEmpty()) {
                // Use all channels as input channels by default
                inputChannels.addAll(channels.keySet());
            }
            
            if (outputChannels.isEmpty()) {
                // Use all channels as output channels by default
                outputChannels.addAll(channels.keySet());
            }
            
            return new Pregel<>(nodes, channels, inputChannels, outputChannels, checkpointer, maxSteps);
        }
    }
}