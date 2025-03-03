# LangGraph Java

A Java implementation of the [LangGraph](https://github.com/langchain-ai/langgraph) framework for building stateful, streaming LLM applications.

## Overview

LangGraph Java is designed for building directed, stateful computational graphs suitable for orchestrating LLM-based applications. The framework is particularly useful for:

- Building agents with tools, memory, and planning abilities
- Creating multi-agent systems with communication channels
- Implementing retrieval augmented generation (RAG) pipelines
- Supporting streaming output for responsive UI experiences

Key features:

- **Type-safe execution** with Java generics
- **Stateful graph execution** with checkpoint persistence
- **Streaming output** for real-time feedback
- **Directed computation graphs** with deterministic execution

## Project Structure

- `langgraph-checkpoint`: Base persistence interfaces
- `langgraph-core`: Main library with channels, Pregel implementation
- `langgraph-examples`: Example applications

## Requirements

- Java 17 or higher
- Gradle 7.0 or higher

## Building

```bash
./gradlew build
```

## Getting Started

### Basic Example

Here's a simple example that creates a graph with a single node that adds 1 to its input:

```java
import com.langgraph.channels.LastValue;
import com.langgraph.pregel.Pregel;
import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;

import java.util.HashMap;
import java.util.Map;

public class SimpleExample {
    public static void main(String[] args) {
        // Create a node that adds 1 to the input
        PregelNode<Integer, Integer> node = new PregelNode.Builder<>("adder", 
            new PregelExecutable<Integer, Integer>() {
                @Override
                public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                    // Get input value, default to 0 if not present
                    int inputValue = inputs.getOrDefault("input", 0);
                    
                    // Return output with value increased by 1
                    Map<String, Integer> output = new HashMap<>();
                    output.put("output", inputValue + 1);
                    return output;
                }
            })
            .channels("input")        // Read from "input" channel
            .triggerChannels("input") // Triggered by "input" updates
            .writers("output")        // Write to "output" channel
            .build();
        
        // Create channels
        Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
        channels.put("input", LastValue.<Integer>create("input"));
        channels.put("output", LastValue.<Integer>create("output"));
        
        // Create Pregel instance
        Pregel<Integer, Integer> pregel = new Pregel.Builder<Integer, Integer>()
                .addNode(node)
                .addChannels(channels)
                .build();
        
        // Run with input 5
        Map<String, Integer> input = new HashMap<>();
        input.put("input", 5);
        Map<String, Integer> result = pregel.invoke(input, null);
        
        // Print result (should be 6)
        System.out.println("Result: " + result.get("output"));
    }
}
```

### Multi-Step Graph Example

Here's an example of a two-node graph that performs sequential processing:

```java
import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.pregel.Pregel;
import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;

import java.util.*;

public class SequentialExample {
    public static void main(String[] args) {
        // First node: Add 1 to the input and write to intermediate channel
        PregelNode<Integer, Integer> adder = new PregelNode.Builder<>("adder", 
            new PregelExecutable<Integer, Integer>() {
                @Override
                public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                    int inputValue = inputs.getOrDefault("input", 0);
                    System.out.println("Adder received input: " + inputValue);
                    
                    // Add 1 to the input value
                    int result = inputValue + 1;
                    
                    // Write to the intermediate channel "state"
                    Map<String, Integer> output = new HashMap<>();
                    output.put("state", result);
                    return output;
                }
            })
            .channels("input")
            .triggerChannels("input")
            .writers("state")
            .build();
        
        // Second node: Multiply intermediate value by 2 and write to output
        PregelNode<Integer, Integer> multiplier = new PregelNode.Builder<>("multiplier", 
            new PregelExecutable<Integer, Integer>() {
                @Override
                public Map<String, Integer> execute(Map<String, Integer> inputs, Map<String, Object> context) {
                    // Get state value, default to 1 if not present
                    int stateValue = inputs.getOrDefault("state", 1);
                    
                    // Multiply by 2
                    int result = stateValue * 2;
                    
                    // Write to the output channel
                    Map<String, Integer> output = new HashMap<>();
                    output.put("output", result);
                    return output;
                }
            })
            .channels("state")
            .triggerChannels("state")
            .writers("output")
            .build();
        
        // Create and configure channels
        Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
        channels.put("input", LastValue.<Integer>create("input"));
        channels.put("state", LastValue.<Integer>create("state"));
        channels.put("output", LastValue.<Integer>create("output"));
        
        // Create Pregel instance with both nodes
        Pregel<Integer, Integer> pregel = new Pregel.Builder<Integer, Integer>()
                .addNode(adder)
                .addNode(multiplier)
                .addChannels(channels)
                .build();
        
        // Run with input 5
        Map<String, Integer> input = Collections.singletonMap("input", 5);
        Map<String, Integer> result = pregel.invoke(input, null);
        
        // Print result: (5 + 1) * 2 = 12
        System.out.println("Result: " + result.get("output"));
    }
}
```

## Advanced Usage

### Working with String Data

```java
// Create a node that processes string data
PregelNode<String, String> processor = new PregelNode.Builder<>("processor", 
    new PregelExecutable<String, String>() {
        @Override
        public Map<String, String> execute(Map<String, String> inputs, Map<String, Object> context) {
            String input = inputs.getOrDefault("input", "");
            Map<String, String> output = new HashMap<>();
            output.put("output", input.toUpperCase());
            return output;
        }
    })
    .channels("input")
    .triggerChannels("input")
    .writers("output")
    .build();

// Create channels
Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
channels.put("input", LastValue.<String>create("input"));
channels.put("output", LastValue.<String>create("output"));

// Create Pregel instance
Pregel<String, String> pregel = new Pregel.Builder<String, String>()
        .addNode(processor)
        .addChannels(channels)
        .build();
```

### Working with JSON-like Data

```java
// Create a node that processes Map<String, Object> data (JSON-like)
PregelNode<Map<String, Object>, Map<String, Object>> processor = 
    new PregelNode.Builder<>("processor", 
        new PregelExecutable<Map<String, Object>, Map<String, Object>>() {
            @Override
            public Map<String, Map<String, Object>> execute(
                    Map<String, Map<String, Object>> inputs, 
                    Map<String, Object> context) {
                
                Map<String, Object> input = inputs.getOrDefault("input", Collections.emptyMap());
                
                // Process input
                Map<String, Object> result = new HashMap<>(input);
                result.put("processed", true);
                
                Map<String, Map<String, Object>> output = new HashMap<>();
                output.put("output", result);
                return output;
            }
        })
        .channels("input")
        .triggerChannels("input")
        .writers("output")
        .build();

// Create channels
Map<String, BaseChannel<?, ?, ?>> channels = new HashMap<>();
channels.put("input", LastValue.<Map<String, Object>>create("input"));
channels.put("output", LastValue.<Map<String, Object>>create("output"));

// Create Pregel instance
Pregel<Map<String, Object>, Map<String, Object>> pregel = 
    new Pregel.Builder<Map<String, Object>, Map<String, Object>>()
        .addNode(processor)
        .addChannels(channels)
        .build();
```

## Channel Types

LangGraph Java provides different channel types for different use cases:

- **LastValue**: Stores the last value written to the channel
- **TopicChannel**: Collects multiple values into a list
- **EphemeralValue**: Only available for the current execution step

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.