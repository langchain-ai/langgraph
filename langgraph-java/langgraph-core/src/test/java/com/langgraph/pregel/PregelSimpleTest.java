package com.langgraph.pregel;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.channels.TopicChannel;
import com.langgraph.pregel.registry.ChannelRegistry;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

public class PregelSimpleTest {

    /**
     * Test a very basic topic channel to understand its behavior with type-safe nodes
     */
    @Test
    void testBasicTopicChannel() {
        System.out.println("\n\n==== RUNNING testBasicTopicChannel ====");
        // Create two nodes that both write to the same TopicChannel
        
        // First node returns a fixed value (111) to output
        PregelNode<Integer, Integer> one = new PregelNode.Builder<>("one", 
            new PregelExecutable<Integer, Integer>() {
                @Override
                public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                    Map<String, Integer> output = new HashMap<>();
                    output.put("output", 111);
                    return output;
                }
            })
        .channels("input")       // Read from input channel
        .triggerChannels("input") // Add trigger for Python compatibility
        .writers("output")       // Write to output channel 
        .build();
        
        // Second node returns a fixed value (222) to output
        PregelNode<Integer, Integer> two = new PregelNode.Builder<>("two", 
            new PregelExecutable<Integer, Integer>() {
                @Override
                public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                    Map<String, Integer> output = new HashMap<>();
                    output.put("output", 222);
                    return output;
                }
            })
        .channels("input")       // Read from input channel
        .triggerChannels("input") // Add trigger for Python compatibility
        .writers("output")       // Write to output channel
        .build();
        
        // Setup channels with TopicChannel for output to collect multiple values
        LastValue<Integer> inputChannel = LastValue.<Integer>create("input");
        TopicChannel<Integer> outputChannel = TopicChannel.<Integer>create();
        
        // No need to initialize input channel (Python-compatible)
        
        Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
        channels.put("input", inputChannel);
        channels.put("output", outputChannel);
        
        // Create a type-safe Pregel instance with both nodes and channels
        Pregel<Integer, Object> pregel = new Pregel.Builder<Integer, Object>()
                .addNode(one)
                .addNode(two)
                .addChannels(channels)
                .build();
        
        // Provide initial input with correct types
        Map<String, Integer> input = new HashMap<>();
        input.put("input", 0);
        
        // Invoke Pregel with type safety
        System.out.println("Invoking Pregel...");
        System.out.println("One: " + one);
        System.out.println("Two: " + two);
        System.out.println("Is one's output channel the same as the output channel? " + outputChannel.equals(pregel.getChannel("output")));
        System.out.println("Is two's output channel the same as the output channel? " + outputChannel.equals(pregel.getChannel("output")));
        Map<String, Object> result = pregel.invoke(input, null);
        System.out.println("Result: " + result);
        
        // Debug the output channel state
        System.out.println("Output channel state:");
        System.out.println("  Class: " + outputChannel.getClass().getName());
        System.out.println("  Type: " + outputChannel.getValueType().getName());
        System.out.println("  Update type: " + outputChannel.getUpdateType().getName());
        System.out.println("  Checkpoint type: " + outputChannel.getCheckpointType().getName());
        
        // Debug full output
        Map<?, ?> map = result;
        System.out.println("Map size: " + map.size());
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            System.out.println("  " + entry.getKey() + " = " + entry.getValue() + 
                " (" + (entry.getValue() != null ? entry.getValue().getClass().getName() : "null") + ")");
        }
        
        // Make assertions with more detailed diagnostics if they fail
        try {
            // Check for the output key
            assertThat(result).containsKey("output");
            
            // The output should be a list
            Object outputValue = result.get("output");
            assertThat(outputValue).isInstanceOf(List.class);
            
            @SuppressWarnings("unchecked")
            List<Integer> outputList = (List<Integer>) outputValue;
            
            // With our enhanced TopicChannel implementation that correctly handles
            // single-value updates, both node values should be in the list
            assertThat(outputList).hasSize(2);
            assertThat(outputList).contains(111, 222);
        } catch (AssertionError e) {
            System.err.println("Assertion failed:");
            System.err.println("Actual result: " + result);
            throw e;
        }
    }
    
    /**
     * Test the enhanced TopicChannel with explicit single-value updates
     */
    @Test
    void testExplicitTopicChannelSingleValueUpdates() {
        // Create a topic channel
        TopicChannel<Integer> channel = TopicChannel.<Integer>create();
        
        // Update with single values
        channel.updateSingleValue(10);
        channel.updateSingleValue(20);
        
        // Test via the channel registry too
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("numbers", channel);
        
        // Update via registry methods
        registry.update("numbers", 30);
        registry.update("numbers", 40);
        
        // Verify all values are accumulated
        List<Integer> values = channel.get();
        assertThat(values).hasSize(4);
        assertThat(values).containsExactly(10, 20, 30, 40);
    }
}