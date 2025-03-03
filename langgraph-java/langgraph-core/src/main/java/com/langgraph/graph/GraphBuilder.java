package com.langgraph.graph;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.channels.TopicChannel;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.pregel.Pregel;
import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;
import com.langgraph.pregel.retry.RetryPolicy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * A fluent builder for creating type-safe computational graphs.
 * Provides a convenient interface for constructing complex graphs with
 * properly typed nodes and channels.
 *
 * @param <I> The input type for the graph
 * @param <O> The output type for the graph
 */
public class GraphBuilder<I, O> {
    
    private final Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
    private final Map<String, PregelNode<I, O>> nodes = new HashMap<>();
    private BaseCheckpointSaver checkpointer;
    private int maxSteps = 100;
    
    /**
     * Creates a new GraphBuilder with the specified input and output types.
     * 
     * @param <I> Input type for the graph
     * @param <O> Output type for the graph
     * @return A new GraphBuilder instance
     */
    public static <I, O> GraphBuilder<I, O> create() {
        return new GraphBuilder<>();
    }
    
    /**
     * Creates a new GraphBuilder with String input and output types.
     * This is a convenience method for creating graphs that work with String data.
     * 
     * @return A new GraphBuilder instance with String input and output types
     */
    public static GraphBuilder<String, String> createStringGraph() {
        return new GraphBuilder<>();
    }
    
    /**
     * Creates a new GraphBuilder with Map input and output types.
     * This is a convenience method for creating graphs that work with JSON-like data.
     * 
     * @return A new GraphBuilder instance with Map input and output types
     */
    public static GraphBuilder<Map<String, Object>, Map<String, Object>> createJsonGraph() {
        return new GraphBuilder<>();
    }
    
    /**
     * Adds a node to the graph with the specified name and executable.
     * By default, the node is configured to read from "input" and write to "output",
     * with "input" as its trigger channel.
     * 
     * @param name The name of the node
     * @param executable The executable for the node
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> addNode(String name, PregelExecutable<I, O> executable) {
        PregelNode<I, O> node = new PregelNode.Builder<I, O>(name, executable)
                .channels("input")
                .triggerChannels("input")
                .writers("output")
                .build();
        nodes.put(name, node);
        return this;
    }
    
    /**
     * Adds a node to the graph with the specified name, executable, and configuration.
     * The configurator is a function that can be used to customize the node builder.
     * 
     * @param name The name of the node
     * @param executable The executable for the node
     * @param configurator A function that configures the node builder
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> addNode(String name, PregelExecutable<I, O> executable, 
                                     Function<PregelNode.Builder<I, O>, PregelNode.Builder<I, O>> configurator) {
        PregelNode.Builder<I, O> builder = new PregelNode.Builder<>(name, executable);
        builder = configurator.apply(builder);
        nodes.put(name, builder.build());
        return this;
    }
    
    /**
     * Adds a pre-built node to the graph.
     * 
     * @param node The node to add
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> addNode(PregelNode<I, O> node) {
        nodes.put(node.getName(), node);
        return this;
    }
    
    /**
     * Adds a LastValue channel to the graph.
     * 
     * @param name The name of the channel
     * @param <T> The type of the value stored in the channel
     * @return This builder for method chaining
     */
    public <T> GraphBuilder<I, O> addLastValueChannel(String name) {
        LastValue<T> channel = LastValue.<T>create(name);
        channels.put(name, channel);
        return this;
    }
    
    /**
     * Adds a TopicChannel to the graph.
     * 
     * @param name The name of the channel
     * @param <T> The type of the value stored in the channel
     * @return This builder for method chaining
     */
    public <T> GraphBuilder<I, O> addTopicChannel(String name) {
        TopicChannel<T> channel = TopicChannel.<T>create(name);
        channels.put(name, channel);
        return this;
    }
    
    /**
     * Adds a custom channel to the graph.
     * 
     * @param name The name of the channel
     * @param channel The channel to add
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> addChannel(String name, BaseChannel<?, ?, ?> channel) {
        channels.put(name, channel);
        return this;
    }
    
    /**
     * Sets the checkpoint saver for the graph.
     * 
     * @param checkpointer The checkpoint saver to use
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> setCheckpointer(BaseCheckpointSaver checkpointer) {
        this.checkpointer = checkpointer;
        return this;
    }
    
    /**
     * Sets the maximum number of steps for the graph.
     * 
     * @param maxSteps The maximum number of steps
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
        return this;
    }
    
    /**
     * Sets the retry policy for all nodes in the graph.
     * Note: This creates new nodes with the specified retry policy.
     * 
     * @param retryPolicy The retry policy to use
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> setRetryPolicy(RetryPolicy retryPolicy) {
        // We need to recreate the nodes with the new retry policy
        Map<String, PregelNode<I, O>> updatedNodes = new HashMap<>();
        
        for (Map.Entry<String, PregelNode<I, O>> entry : nodes.entrySet()) {
            String nodeName = entry.getKey();
            PregelNode<I, O> node = entry.getValue();
            
            // Create a new node with the same configuration but different retry policy
            PregelNode<I, O> updatedNode = new PregelNode<>(
                node.getName(), 
                node.getAction(),
                node.getChannels(),
                node.getTriggerChannels(),
                node.getWriteEntries(),
                retryPolicy
            );
            
            updatedNodes.put(nodeName, updatedNode);
        }
        
        // Replace all nodes with updated ones
        nodes.clear();
        nodes.putAll(updatedNodes);
        
        return this;
    }
    
    /**
     * Configures the node channels to form an implied sequence.
     * This creates a chain of nodes where the output of one node is the input of the next.
     * 
     * @param nodeNames The names of the nodes in the sequence
     * @param inputChannel The name of the input channel
     * @param outputChannel The name of the output channel
     * @param intermediateChannel The name of the intermediate channel
     * @return This builder for method chaining
     */
    public GraphBuilder<I, O> configureSequence(List<String> nodeNames, String inputChannel, 
                                              String outputChannel, String intermediateChannel) {
        if (nodeNames.size() < 2) {
            throw new IllegalArgumentException("Sequence must have at least 2 nodes");
        }
        
        // First node reads from input, writes to intermediate
        PregelNode<I, O> firstNode = nodes.get(nodeNames.get(0));
        PregelNode.Builder<I, O> builder = new PregelNode.Builder<>(firstNode.getName(), firstNode.getAction())
                .channels(inputChannel)
                .triggerChannels(inputChannel)
                .writers(intermediateChannel);
        nodes.put(firstNode.getName(), builder.build());
        
        // Middle nodes read from intermediate, write to intermediate
        for (int i = 1; i < nodeNames.size() - 1; i++) {
            PregelNode<I, O> node = nodes.get(nodeNames.get(i));
            builder = new PregelNode.Builder<>(node.getName(), node.getAction())
                    .channels(intermediateChannel)
                    .triggerChannels(intermediateChannel)
                    .writers(intermediateChannel);
            nodes.put(node.getName(), builder.build());
        }
        
        // Last node reads from intermediate, writes to output
        PregelNode<I, O> lastNode = nodes.get(nodeNames.get(nodeNames.size() - 1));
        builder = new PregelNode.Builder<>(lastNode.getName(), lastNode.getAction())
                .channels(intermediateChannel)
                .triggerChannels(intermediateChannel)
                .writers(outputChannel);
        nodes.put(lastNode.getName(), builder.build());
        
        return this;
    }
    
    /**
     * Builds a Pregel graph with the configured nodes and channels.
     * 
     * @return A new Pregel instance
     */
    public Pregel<I, O> build() {
        // If we didn't add the input and output channels explicitly, add them
        if (!channels.containsKey("input")) {
            addLastValueChannel("input");
        }
        
        if (!channels.containsKey("output")) {
            addLastValueChannel("output");
        }
        
        return new Pregel.Builder<I, O>()
                .addNodes(new ArrayList<>(nodes.values()))
                .addChannels(channels)
                .setCheckpointer(checkpointer)
                .setMaxSteps(maxSteps)
                .build();
    }
}