package com.langgraph.graph;

import com.langgraph.pregel.Pregel;
import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;

public class GraphBuilderTest {

    /**
     * Simple executable that adds one to the input value
     */
    private static class AddOneExecutable implements PregelExecutable<Integer, Integer> {
        @Override
        public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
            // Get input value, default to 0 if not present
            int inputValue = inputs.getOrDefault("input", 0);
            
            // Return output with value increased by 1
            Map<String, Integer> output = new HashMap<>();
            output.put("output", inputValue + 1);
            return output;
        }
    }
    
    /**
     * Executable that multiplies input by a factor
     */
    private static class MultiplyExecutable implements PregelExecutable<Integer, Integer> {
        private final int factor;
        
        public MultiplyExecutable(int factor) {
            this.factor = factor;
        }
        
        @Override
        public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
            // Get input value, default to 1 if not present
            int inputValue = inputs.getOrDefault("state", 1);
            
            // Return output with value multiplied by factor
            Map<String, Integer> output = new HashMap<>();
            output.put("output", inputValue * factor);
            return output;
        }
    }
    
    /**
     * Test creating a simple graph with the builder
     */
    @Test
    void testBasicGraphBuilder() {
        // Create a graph with a single node
        Pregel<Integer, Integer> graph = GraphBuilder.<Integer, Integer>create()
                .addNode("adder", new AddOneExecutable())
                .build();
        
        // Run the graph with input 5
        Map<String, Integer> input = Collections.singletonMap("input", 5);
        Map<String, Integer> result = graph.invoke(input, null);
        
        // Verify the result
        assertThat(result).containsEntry("output", 6);
    }
    
    /**
     * Test creating a sequence of nodes with the builder
     */
    @Test
    void testSequenceGraphBuilder() {
        // Create a graph with multiple nodes in sequence
        GraphBuilder<Integer, Integer> builder = GraphBuilder.create();
        
        // Add nodes with specialized executables
        // The adder will read from "input" and write to "state"
        // The multiplier will read from "state" and write to "output"
        builder.addNode("adder", new PregelExecutable<Integer, Integer>() {
            @Override
            public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                // Get input value, default to 0 if not present
                int inputValue = inputs.getOrDefault("input", 0);
                System.out.println("Adder received input: " + inputValue);
                
                // Add 1 to the input value
                int result = inputValue + 1;
                System.out.println("Adder result: " + result);
                
                // Write to the intermediate channel "state"
                Map<String, Integer> output = new HashMap<>();
                output.put("state", result);
                return output;
            }
        });
        
        builder.addNode("multiplier", new PregelExecutable<Integer, Integer>() {
            @Override
            public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                // Get state value, default to 1 if not present
                int stateValue = inputs.getOrDefault("state", 1);
                System.out.println("Multiplier received state: " + stateValue);
                
                // Multiply by 2
                int result = stateValue * 2;
                System.out.println("Multiplier result: " + result);
                
                // Write to the output channel
                Map<String, Integer> output = new HashMap<>();
                output.put("output", result);
                return output;
            }
        });
        
        // Set up channels - we need input, output, and an intermediate channel
        builder.addLastValueChannel("input");
        builder.addLastValueChannel("state");
        builder.addLastValueChannel("output");
        
        // Configure the sequence: input -> adder -> multiplier -> output
        List<String> nodeSequence = Arrays.asList("adder", "multiplier");
        builder.configureSequence(nodeSequence, "input", "output", "state");
        
        // Build the graph
        Pregel<Integer, Integer> graph = builder.build();
        
        // Print out the graph structure for debugging
        System.out.println("Graph nodes:");
        for (String name : graph.getNodeRegistry().getAll().keySet()) {
            PregelNode<?, ?> node = graph.getNodeRegistry().get(name);
            System.out.println("  Node: " + node.getName());
            System.out.println("    Channels: " + node.getChannels());
            System.out.println("    Triggers: " + node.getTriggerChannels());
            System.out.println("    Writers: " + node.getWriters());
        }
        
        // Run the graph with input 5
        Map<String, Integer> input = Collections.singletonMap("input", 5);
        System.out.println("Running graph with input: " + input);
        Map<String, Integer> result = graph.invoke(input, null);
        
        // Print result for debugging
        System.out.println("Result: " + result);
        
        // Verify the result: (5 + 1) * 2 = 12
        assertThat(result).containsEntry("output", 12);
    }
    
    /**
     * Test creating a graph with custom node configuration
     */
    @Test
    void testCustomNodeConfiguration() {
        // Create a graph with a node that has custom configuration
        Pregel<Integer, Integer> graph = GraphBuilder.<Integer, Integer>create()
                .addNode("adder", new AddOneExecutable(), builder -> 
                    builder.channels("input").channels("extra")
                           .triggerChannels("input")
                           .writers("output", "debug"))
                .addLastValueChannel("extra")
                .addLastValueChannel("debug")
                .build();
        
        // Verify the graph was created successfully
        assertThat(graph).isNotNull();
        
        // Run the graph with input 5
        Map<String, Integer> input = Collections.singletonMap("input", 5);
        Map<String, Integer> result = graph.invoke(input, null);
        
        // Verify the result
        assertThat(result).containsEntry("output", 6);
    }
    
    /**
     * Test creating a graph with string input/output
     */
    @Test
    void testStringGraph() {
        // Create a string graph with a simple node
        Pregel<String, String> graph = GraphBuilder.<String, String>createStringGraph()
                .addNode("echo", new PregelExecutable<String, String>() {
                    @Override
                    public Map<String, String> execute(Map<String, String> inputs, Map<String, Object> context) {
                        String input = inputs.getOrDefault("input", "");
                        Map<String, String> output = new HashMap<>();
                        output.put("output", input.toUpperCase());
                        return output;
                    }
                })
                .build();
        
        // Run the graph with input "hello"
        Map<String, String> input = Collections.singletonMap("input", "hello");
        Map<String, String> result = graph.invoke(input, null);
        
        // Verify the result
        assertThat(result).containsEntry("output", "HELLO");
    }
    
    /**
     * Test creating a JSON-like graph
     */
    @Test
    void testJsonGraph() {
        // Create a JSON graph with a simple node
        Pregel<Map<String, Object>, Map<String, Object>> graph = GraphBuilder.createJsonGraph()
                .addNode("processor", new PregelExecutable<Map<String, Object>, Map<String, Object>>() {
                    @Override
                    public Map<String, Map<String, Object>> execute(
                            Map<String, Map<String, Object>> inputs, 
                            Map<String, Object> context) {
                        
                        Map<String, Object> input = inputs.getOrDefault("input", Collections.emptyMap());
                        
                        // Create a result with modified input
                        Map<String, Object> result = new HashMap<>(input);
                        result.put("processed", true);
                        
                        Map<String, Map<String, Object>> output = new HashMap<>();
                        output.put("output", result);
                        return output;
                    }
                })
                .build();
        
        // Create input with some JSON-like data
        Map<String, Object> jsonData = new HashMap<>();
        jsonData.put("name", "test");
        jsonData.put("value", 123);
        
        Map<String, Map<String, Object>> input = Collections.singletonMap("input", jsonData);
        
        // Run the graph
        Map<String, Map<String, Object>> result = graph.invoke(input, null);
        
        // Verify the result
        assertThat(result).containsKey("output");
        Map<String, Object> outputData = result.get("output");
        assertThat(outputData).containsEntry("name", "test");
        assertThat(outputData).containsEntry("value", 123);
        assertThat(outputData).containsEntry("processed", true);
    }
}