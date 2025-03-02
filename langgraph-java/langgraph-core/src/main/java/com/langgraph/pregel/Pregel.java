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
    
    /**
     * Create a Pregel instance with all parameters.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     * @param checkpointer Optional checkpointer for persisting state
     * @param maxSteps Maximum number of steps to execute
     */
    public Pregel(
            Map<String, PregelNode> nodes,
            Map<String, BaseChannel> channels,
            BaseCheckpointSaver checkpointer,
            int maxSteps) {
        // Initialize registries
        this.nodeRegistry = new NodeRegistry(nodes);
        this.channelRegistry = new ChannelRegistry(channels);
        this.checkpointer = checkpointer;
        this.executor = Executors.newWorkStealingPool();
        this.maxSteps = maxSteps;
        
        // Validate configuration
        validate();
    }
    
    /**
     * Create a Pregel instance with default max steps.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     * @param checkpointer Optional checkpointer for persisting state
     */
    public Pregel(
            Map<String, PregelNode> nodes,
            Map<String, BaseChannel> channels,
            BaseCheckpointSaver checkpointer) {
        this(nodes, channels, checkpointer, 100);
    }
    
    /**
     * Create a Pregel instance without checkpointing.
     *
     * @param nodes Map of node names to nodes
     * @param channels Map of channel names to channels
     */
    public Pregel(Map<String, PregelNode> nodes, Map<String, BaseChannel> channels) {
        this(nodes, channels, null);
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
        return pregelLoop.execute(inputMap, context, threadId);
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
        
        @SuppressWarnings("unchecked")
        Map<String, Object> stateMap = (Map<String, Object>) state;
        
        // Update channels with the state
        initializeChannels(stateMap);
        
        // Create a checkpoint
        if (checkpointer != null) {
            checkpointer.checkpoint(threadId, stateMap);
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
     * Initialize channels with input.
     *
     * @param input Input map
     */
    private void initializeChannels(Map<String, Object> input) {
        if (input == null || input.isEmpty()) {
            return;
        }
        
        // Update channels with input values
        channelRegistry.updateAll(input);
    }
    
    /**
     * Convert input to a map if necessary.
     *
     * @param input Input object
     * @return Input as a map
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> convertInput(Object input) {
        if (input == null) {
            return Collections.emptyMap();
        }
        
        if (input instanceof Map) {
            return (Map<String, Object>) input;
        }
        
        // Handle special cases or throw exception
        throw new IllegalArgumentException("Input must be a Map<String, Object>");
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
            return new Pregel(nodes, channels, checkpointer, maxSteps);
        }
    }
}