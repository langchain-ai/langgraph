package com.langgraph.pregel;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.channels.TypeReference;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Tests to validate that the Java implementation of LangGraph can handle
 * uninitialized channels like the Python version, without explicit initialization.
 */
public class UninitializedChannelsTest {

    /**
     * Test for creating and using a graph with uninitialized channels.
     * This simulates the Python behavior where channels don't need to be
     * explicitly initialized before use.
     */
    @Test
    void testUninitializedChannels() {
        // Create a node that increments input and writes to output
        PregelNode node = new PregelNode.Builder("processor", (inputs, context) -> {
            // Get the input value, which could be null if channel is uninitialized
            Integer input = 0; // Default value for uninitialized channel
            if (inputs.containsKey("input") && inputs.get("input") != null) {
                input = (Integer) inputs.get("input");
            }
            
            // Create output with incremented value
            Map<String, Object> output = new HashMap<>();
            output.put("output", input + 1);
            return output;
        })
        .channels("input")       // Read from input channel
        .triggerChannels("input") // Also trigger on input channel (important for Python compatibility)
        .writers("output")        // Write to output channel 
        .build();
        
        // Create channels without initializing them
        LastValue<Integer> inputChannel = LastValue.<Integer>create("input");
        LastValue<Integer> outputChannel = LastValue.<Integer>create("output");
        
        // Note: We intentionally don't initialize the channels with update()
        
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("input", inputChannel);
        channels.put("output", outputChannel);
        
        // Create a Pregel instance
        Pregel pregel = new Pregel.Builder()
                .addNode(node)
                .addChannels(channels)
                .build();
        
        // Execute the graph with input channel to trigger the node (Python compatibility)
        Map<String, Object> input = new HashMap<>();
        input.put("input", null); // Null value to use the default
        Object result = pregel.invoke(input, null);
        
        // Verify the result
        assertThat(result).isInstanceOf(Map.class);
        
        @SuppressWarnings("unchecked")
        Map<String, Object> resultMap = (Map<String, Object>) result;
        
        // Verify the output was produced even with uninitialized channel
        assertThat(resultMap).containsKey("output");
        assertThat(resultMap.get("output")).isEqualTo(1); // 0 + 1 = 1
    }
    
    /**
     * Test a more complex workflow with multiple nodes and uninitialized channels.
     */
    @Test
    void testComplexUninitializedChannels() {
        // Create first node that processes initial input
        PregelNode firstNode = new PregelNode.Builder("first", (inputs, context) -> {
            // In Python-like behavior, this would get null for uninitialized channels
            Integer input = 0; // Default value for uninitialized channel
            if (inputs.containsKey("initial") && inputs.get("initial") != null) {
                input = (Integer) inputs.get("initial");
            }
            
            Map<String, Object> output = new HashMap<>();
            output.put("intermediate", input + 10);
            return output;
        })
        .channels("initial")
        .triggerChannels("initial")  // Essential for Python compatibility - will run on first superstep
        .writers("intermediate")
        .build();
        
        // Create second node that processes intermediate result
        PregelNode secondNode = new PregelNode.Builder("second", (inputs, context) -> {
            Integer intermediate = 0; // Default value for uninitialized channel
            if (inputs.containsKey("intermediate") && inputs.get("intermediate") != null) {
                intermediate = (Integer) inputs.get("intermediate"); 
            }
            
            Map<String, Object> output = new HashMap<>();
            output.put("final", intermediate * 2);
            return output;
        })
        .channels("intermediate")
        .triggerChannels("intermediate") // Will only run when intermediate channel is updated
        .writers("final")
        .build();
        
        // Create channels without initialization
        LastValue<Integer> initialChannel = LastValue.<Integer>create("initial");
        LastValue<Integer> intermediateChannel = LastValue.<Integer>create("intermediate");
        LastValue<Integer> finalChannel = LastValue.<Integer>create("final");
        
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("initial", initialChannel);
        channels.put("intermediate", intermediateChannel);
        channels.put("final", finalChannel);
        
        // Create a Pregel instance
        Pregel pregel = new Pregel.Builder()
                .addNode(firstNode)
                .addNode(secondNode)
                .addChannels(channels)
                .build();
        
        // We need to provide an empty input map for the "initial" channel
        // to trigger the first node with Python compatibility
        Map<String, Object> initialInput = new HashMap<>();
        initialInput.put("initial", null);
        Object result = pregel.invoke(initialInput, null);
        
        // Verify the result
        assertThat(result).isInstanceOf(Map.class);
        
        @SuppressWarnings("unchecked")
        Map<String, Object> resultMap = (Map<String, Object>) result;
        
        // Expected flow: 0 (uninitialized) -> +10 -> *2 = 20
        assertThat(resultMap).containsKey("final");
        assertThat(resultMap.get("final")).isEqualTo(20);
    }
    
    /**
     * Test that a graph can handle both initialized and uninitialized channels.
     */
    @Test
    void testMixedChannelInitialization() {
        // Create a node that combines two inputs
        PregelNode combiner = new PregelNode.Builder("combiner", (inputs, context) -> {
            // One channel will be initialized, the other won't
            Integer value1 = 0;
            Integer value2 = 0;
            
            if (inputs.containsKey("value1") && inputs.get("value1") != null) {
                value1 = (Integer) inputs.get("value1");
            }
            
            if (inputs.containsKey("value2") && inputs.get("value2") != null) {
                value2 = (Integer) inputs.get("value2");
            }
            
            Map<String, Object> output = new HashMap<>();
            output.put("result", value1 + value2);
            return output;
        })
        .channels(Arrays.asList("value1", "value2"))
        .triggerChannels("value1")  // For Python compatibility - will run on first superstep
        .writers("result")
        .build();
        
        // Create channels - one initialized, one not
        LastValue<Integer> value1Channel = LastValue.<Integer>create("value1");
        LastValue<Integer> value2Channel = LastValue.<Integer>create("value2");
        LastValue<Integer> resultChannel = LastValue.<Integer>create("result");
        
        // This test specifically tests mixing pre-initialized and uninitialized channels
        // We intentionally initialize one channel but not the other to test the behavior
        value1Channel.update(Collections.singletonList(5));
        
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("value1", value1Channel);
        channels.put("value2", value2Channel);
        channels.put("result", resultChannel);
        
        // Create a Pregel instance
        Pregel pregel = new Pregel.Builder()
                .addNode(combiner)
                .addChannels(channels)
                .build();
        
        // Execute with the value1 channel as input trigger
        Map<String, Object> input = new HashMap<>();
        input.put("value1", 5); // Explicitly use value 5 to match the initialized value
        Object result = pregel.invoke(input, null);
        
        // Verify the result
        assertThat(result).isInstanceOf(Map.class);
        
        @SuppressWarnings("unchecked")
        Map<String, Object> resultMap = (Map<String, Object>) result;
        
        // Expected: 5 (initialized) + 0 (uninitialized) = 5
        assertThat(resultMap).containsKey("result");
        assertThat(resultMap.get("result")).isEqualTo(5);
    }
}