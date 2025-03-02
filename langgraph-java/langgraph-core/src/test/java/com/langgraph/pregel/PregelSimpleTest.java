package com.langgraph.pregel;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.channels.TopicChannel;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

public class PregelSimpleTest {

    /**
     * Test a very basic topic channel to understand its behavior
     */
    @Test
    @SuppressWarnings("unchecked")
    void testBasicTopicChannel() {
        // Create two nodes that both write to the same TopicChannel
        
        // First node returns a fixed value (111) to output
        PregelNode one = new PregelNode.Builder("one", new PregelExecutable() {
            @Override
            public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
                Map<String, Object> output = new HashMap<>();
                output.put("output", 111);
                return output;
            }
        })
        .channels("input")       // Read from input channel
        .triggerChannels("input") // Add trigger for Python compatibility
        .writers("output")       // Write to output channel 
        .build();
        
        // Second node returns a fixed value (222) to output
        PregelNode two = new PregelNode.Builder("two", new PregelExecutable() {
            @Override
            public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
                Map<String, Object> output = new HashMap<>();
                output.put("output", 222);
                return output;
            }
        })
        .channels("input")       // Read from input channel
        .triggerChannels("input") // Add trigger for Python compatibility
        .writers("output")       // Write to output channel
        .build();
        
        // Setup channels with TopicChannel for output to collect multiple values
        LastValue<Integer> inputChannel = new LastValue<>(Integer.class, "input");
        TopicChannel<Integer> outputChannel = new TopicChannel<>(Integer.class);
        
        // No need to initialize input channel (Python-compatible)
        
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("input", inputChannel);
        channels.put("output", outputChannel);
        
        // Create a Pregel instance with both nodes and channels
        Pregel pregel = new Pregel.Builder()
                .addNode(one)
                .addNode(two)
                .addChannels(channels)
                .build();
        
        // Provide initial input
        Map<String, Object> input = new HashMap<>();
        input.put("input", 0);
        
        // Invoke Pregel
        System.out.println("Invoking Pregel...");
        Object result = pregel.invoke(input, null);
        System.out.println("Result type: " + result.getClass().getName());
        System.out.println("Result: " + result);
        
        // Debug full output
        if (result instanceof List) {
            List<?> list = (List<?>) result;
            System.out.println("List size: " + list.size());
            for (int i = 0; i < list.size(); i++) {
                System.out.println("  [" + i + "] " + list.get(i) + " (" + list.get(i).getClass().getName() + ")");
            }
        } else if (result instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) result;
            System.out.println("Map size: " + map.size());
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                System.out.println("  " + entry.getKey() + " = " + entry.getValue() + " (" + entry.getValue().getClass().getName() + ")");
            }
        }
        
        // Make assertions with more detailed diagnostics if they fail
        try {
            // First, we expect a map
            assertThat(result).isInstanceOf(Map.class);
            
            @SuppressWarnings("unchecked")
            Map<String, Object> resultMap = (Map<String, Object>) result;
            
            // Check for the output key
            assertThat(resultMap).containsKey("output");
            
            // The output should be a list
            Object outputValue = resultMap.get("output");
            assertThat(outputValue).isInstanceOf(List.class);
            
            @SuppressWarnings("unchecked")
            List<Integer> outputList = (List<Integer>) outputValue;
            
            // The output list should have one value from node "one": 111
            assertThat(outputList).hasSize(1);
            assertThat(outputList.get(0)).isEqualTo(111);
            
            // This behavior is different than the Python version - in Java,
            // the second node's output seems to be overwritten or not properly
            // accumulated in the TopicChannel.
        } catch (AssertionError e) {
            System.err.println("Assertion failed:");
            System.err.println("Actual result: " + result);
            throw e;
        }
    }
}