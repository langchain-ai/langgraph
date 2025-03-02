# LangGraph Java

A Java port of LangGraph, a framework for building stateful, observable applications with large language models (LLMs).

## Project Structure

- `langgraph-checkpoint`: Base persistence interfaces
- `langgraph-core`: Main library with channels, Pregel, and StateGraph
- `langgraph-examples`: Example applications

## Requirements

- Java 17 or higher
- Gradle 7.0 or higher

## Building

```bash
./gradlew build
```

## Features

- Graph-based architecture with nodes and edges
- Type-safe state schema using Java Records
- Cyclical execution patterns
- Checkpoint integration for persistence
- Streaming support
- Human-in-the-loop capabilities

## Usage Example

```java
// Create a graph with our schema
StateGraph<CounterState> graph = new StateGraph<>(CounterState.class);

// Add nodes
graph.addNode("increment", state -> {
    Map<String, Object> updates = new HashMap<>();
    updates.put("count", state.count() + 1);
    return updates;
});

// Add edges
graph.addEdge(START, "increment");
graph.addEdge("increment", "check");

// Add conditional edge
graph.addConditionalEdges("check", state -> {
    if (state.count() >= 3) {
        return "finish";
    }
    return "increment";
});

// Compile and run
CompiledStateGraph<CounterState> compiled = graph.compile();
CounterState result = compiled.invoke(new CounterState(0, ""));
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.