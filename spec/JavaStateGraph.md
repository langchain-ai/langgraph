# Java StateGraph Interfaces

This document defines the Java interfaces for the StateGraph layer of LangGraph, the primary high-level API for building stateful computation graphs.

## Special Constants

Special constants for graph entry and exit points.

```java
package com.langgraph.graph;

/**
 * Special constants for graph construction.
 */
public final class GraphConstants {
    private GraphConstants() {}
    
    /**
     * Special value representing the entry point to the graph.
     */
    public static final String START = "__start__";
    
    /**
     * Special value representing an exit point from the graph.
     */
    public static final String END = "__end__";
}
```

## Interface for Schema Validation

Interface for validating state schemas.

```java
package com.langgraph.graph;

/**
 * Interface for validating state schemas.
 */
public interface SchemaValidator {
    /**
     * Validate a state object against a schema.
     *
     * @param state State to validate
     * @throws IllegalArgumentException if validation fails
     */
    void validate(Object state);
    
    /**
     * Get the schema type.
     *
     * @return Schema class or interface
     */
    Class<?> getSchemaType();
}
```

## Record-based Schema Validator

```java
package com.langgraph.graph;

import java.lang.reflect.Field;
import java.lang.reflect.RecordComponent;
import java.util.Map;

/**
 * Schema validator for Java Record types.
 *
 * @param <T> Record type
 */
public class RecordSchemaValidator<T> implements SchemaValidator {
    private final Class<T> recordClass;
    
    /**
     * Create a validator for a Record class.
     *
     * @param recordClass Record class to validate against
     */
    public RecordSchemaValidator(Class<T> recordClass) {
        if (!recordClass.isRecord()) {
            throw new IllegalArgumentException("Class must be a record: " + recordClass.getName());
        }
        this.recordClass = recordClass;
    }
    
    @Override
    public void validate(Object state) {
        if (state == null) {
            throw new IllegalArgumentException("State cannot be null");
        }
        
        if (!recordClass.isInstance(state)) {
            if (state instanceof Map) {
                // Validate Map against record components
                validateMap((Map<?, ?>) state);
            } else {
                throw new IllegalArgumentException(
                        "State must be an instance of " + recordClass.getName() + 
                        " or a Map with equivalent structure");
            }
        }
    }
    
    @Override
    public Class<?> getSchemaType() {
        return recordClass;
    }
    
    /**
     * Validate a Map against record components.
     *
     * @param stateMap Map to validate
     */
    private void validateMap(Map<?, ?> stateMap) {
        RecordComponent[] components = recordClass.getRecordComponents();
        
        for (RecordComponent component : components) {
            String name = component.getName();
            Class<?> type = component.getType();
            
            if (!stateMap.containsKey(name)) {
                throw new IllegalArgumentException("Missing required field: " + name);
            }
            
            Object value = stateMap.get(name);
            
            // Basic type checking
            if (value != null && !type.isInstance(value)) {
                throw new IllegalArgumentException(
                        "Field '" + name + "' has wrong type. Expected: " + 
                        type.getName() + ", got: " + value.getClass().getName());
            }
        }
    }
}
```

## Node Action Interface

Interface for node actions.

```java
package com.langgraph.graph;

import java.util.Map;

/**
 * Interface for node actions in a graph.
 *
 * @param <S> State type
 */
@FunctionalInterface
public interface NodeAction<S> {
    /**
     * Execute the node action.
     *
     * @param state Current state
     * @return Updates to apply to the state
     */
    Map<String, Object> execute(S state);
}
```

## Edge Condition Interface

Interface for conditional edge routing.

```java
package com.langgraph.graph;

/**
 * Interface for conditional edge routing.
 *
 * @param <S> State type
 */
@FunctionalInterface
public interface EdgeCondition<S> {
    /**
     * Determine the next node based on state.
     *
     * @param state Current state
     * @return Name of the next node
     */
    String route(S state);
}
```

## `StateGraph` Class

The primary class for defining computation graphs.

```java
package com.langgraph.graph;

import static com.langgraph.graph.GraphConstants.END;
import static com.langgraph.graph.GraphConstants.START;

import com.langgraph.channels.Channel;
import com.langgraph.channels.Channels;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.pregel.*;

import java.util.*;
import java.util.function.Function;

/**
 * Main class for defining a computation graph with explicit state.
 *
 * @param <S> State type
 */
public class StateGraph<S> {
    private final SchemaValidator stateValidator;
    private final Map<String, NodeAction<S>> nodes = new LinkedHashMap<>();
    private final Map<String, Set<String>> edges = new LinkedHashMap<>();
    private final Map<String, EdgeCondition<S>> conditionalEdges = new LinkedHashMap<>();
    private final Set<String> finishPoints = new HashSet<>();
    
    private String entryPoint = START;
    private EdgeCondition<S> conditionalEntryPoint;
    
    /**
     * Create a StateGraph with a state schema.
     *
     * @param stateSchema State schema class
     */
    @SuppressWarnings("unchecked")
    public StateGraph(Class<S> stateSchema) {
        if (stateSchema.isRecord()) {
            this.stateValidator = new RecordSchemaValidator<>(stateSchema);
        } else {
            throw new IllegalArgumentException(
                    "State schema must be a record class. Use Java Records for type-safe state schemas.");
        }
    }
    
    /**
     * Add a node to the graph.
     *
     * @param key Node name
     * @param action Node action
     * @return This graph
     */
    public StateGraph<S> addNode(String key, NodeAction<S> action) {
        nodes.put(key, action);
        return this;
    }
    
    /**
     * Add an edge between nodes.
     *
     * @param startKey Starting node
     * @param endKey Ending node
     * @return This graph
     */
    public StateGraph<S> addEdge(String startKey, String endKey) {
        // Special handling for START
        if (!START.equals(startKey) && !nodes.containsKey(startKey)) {
            throw new IllegalArgumentException("Start node not found: " + startKey);
        }
        
        // Special handling for END
        if (!END.equals(endKey) && !nodes.containsKey(endKey)) {
            throw new IllegalArgumentException("End node not found: " + endKey);
        }
        
        edges.computeIfAbsent(startKey, k -> new HashSet<>()).add(endKey);
        
        // Handle END edge
        if (END.equals(endKey)) {
            finishPoints.add(startKey);
        }
        
        return this;
    }
    
    /**
     * Add a sequence of nodes with edges between them.
     *
     * @param nodeKeys Sequence of node keys
     * @return This graph
     */
    public StateGraph<S> addSequence(String... nodeKeys) {
        if (nodeKeys.length < 2) {
            throw new IllegalArgumentException("Sequence must contain at least two nodes");
        }
        
        for (int i = 0; i < nodeKeys.length - 1; i++) {
            addEdge(nodeKeys[i], nodeKeys[i + 1]);
        }
        
        return this;
    }
    
    /**
     * Add conditional edges from a source node.
     *
     * @param source Source node
     * @param condition Condition function
     * @return This graph
     */
    public StateGraph<S> addConditionalEdges(String source, EdgeCondition<S> condition) {
        if (!nodes.containsKey(source)) {
            throw new IllegalArgumentException("Source node not found: " + source);
        }
        
        conditionalEdges.put(source, condition);
        return this;
    }
    
    /**
     * Set an explicit entry point for the graph.
     *
     * @param key Entry point node
     * @return This graph
     */
    public StateGraph<S> setEntryPoint(String key) {
        if (!nodes.containsKey(key)) {
            throw new IllegalArgumentException("Entry point node not found: " + key);
        }
        
        entryPoint = key;
        conditionalEntryPoint = null;
        return this;
    }
    
    /**
     * Set a conditional entry point for the graph.
     *
     * @param condition Condition for determining entry point
     * @return This graph
     */
    public StateGraph<S> setConditionalEntryPoint(EdgeCondition<S> condition) {
        conditionalEntryPoint = condition;
        return this;
    }
    
    /**
     * Set a finish point for the graph.
     *
     * @param key Finish point node
     * @return This graph
     */
    public StateGraph<S> setFinishPoint(String key) {
        if (!nodes.containsKey(key)) {
            throw new IllegalArgumentException("Finish point node not found: " + key);
        }
        
        finishPoints.add(key);
        return this;
    }
    
    /**
     * Validate the graph structure.
     *
     * @return This graph
     */
    public StateGraph<S> validate() {
        // Check that all nodes are connected
        Set<String> reachableNodes = new HashSet<>();
        
        if (conditionalEntryPoint != null) {
            // Can't statically validate conditional entry points
            // We'll assume all nodes could be entry points
            reachableNodes.addAll(nodes.keySet());
        } else {
            // Start from the entry point
            collectReachableNodes(entryPoint, reachableNodes);
        }
        
        // Check for unreachable nodes
        for (String node : nodes.keySet()) {
            if (!reachableNodes.contains(node)) {
                throw new IllegalStateException("Node is unreachable: " + node);
            }
        }
        
        // Check that all nodes have outbound edges or are finish points
        for (String node : nodes.keySet()) {
            boolean hasOutbound = edges.containsKey(node) && !edges.get(node).isEmpty();
            boolean hasConditional = conditionalEdges.containsKey(node);
            boolean isFinish = finishPoints.contains(node);
            
            if (!hasOutbound && !hasConditional && !isFinish) {
                throw new IllegalStateException(
                        "Node has no outbound edges and is not a finish point: " + node);
            }
        }
        
        return this;
    }
    
    /**
     * Recursively collect reachable nodes from a starting point.
     *
     * @param start Starting node
     * @param reachable Set of reachable nodes
     */
    private void collectReachableNodes(String start, Set<String> reachable) {
        if (START.equals(start)) {
            // Special handling for START
            if (nodes.containsKey(entryPoint)) {
                reachable.add(entryPoint);
                collectReachableNodes(entryPoint, reachable);
            }
            return;
        }
        
        if (!nodes.containsKey(start)) {
            return; // Skip special nodes like END
        }
        
        // Mark as reachable
        reachable.add(start);
        
        // Follow static edges
        if (edges.containsKey(start)) {
            for (String next : edges.get(start)) {
                if (!reachable.contains(next) && !END.equals(next)) {
                    collectReachableNodes(next, reachable);
                }
            }
        }
        
        // Can't follow conditional edges statically
        // We'll just ignore them for validation
    }
    
    /**
     * Compile the graph into an executable runnable.
     *
     * @param checkpointer Optional checkpointer for persistence
     * @return Compiled graph
     */
    public CompiledStateGraph<S> compile(BaseCheckpointSaver checkpointer) {
        // Validate the graph
        validate();
        
        // Create the Pregel nodes
        Map<String, PregelNode> pregelNodes = new HashMap<>();
        
        // Add normal nodes
        for (Map.Entry<String, NodeAction<S>> entry : nodes.entrySet()) {
            String nodeName = entry.getKey();
            NodeAction<S> action = entry.getValue();
            
            PregelExecutable executable = createNodeExecutable(action);
            
            Set<String> subscribe = new HashSet<>();
            subscribe.add("state"); // All nodes read from the state channel
            
            Set<String> writers = new HashSet<>();
            writers.add("state"); // All nodes write to the state channel
            writers.add("next"); // All nodes can set the next node
            
            PregelNode node = new PregelNode(
                    nodeName,
                    executable,
                    subscribe,
                    null,
                    writers,
                    null
            );
            
            pregelNodes.put(nodeName, node);
        }
        
        // Add special entry node
        PregelExecutable entryExecutable = createEntryExecutable();
        PregelNode entryNode = new PregelNode(
                "entry",
                entryExecutable,
                Collections.singleton("input"),
                null,
                Collections.singleton("next"),
                null
        );
        pregelNodes.put("entry", entryNode);
        
        // Add router node
        PregelExecutable routerExecutable = createRouterExecutable();
        PregelNode routerNode = new PregelNode(
                "router",
                routerExecutable,
                Collections.singleton("next"),
                null,
                Collections.emptySet(),
                null
        );
        pregelNodes.put("router", routerNode);
        
        // Create channels
        Map<String, Channel> channels = new HashMap<>();
        channels.put("state", Channels.lastValue()); // Main state channel
        channels.put("input", Channels.lastValue()); // Input channel
        channels.put("next", Channels.lastValue()); // Next node channel
        
        // Build the Pregel instance
        Pregel.Builder builder = new Pregel.Builder();
        
        for (Map.Entry<String, PregelNode> entry : pregelNodes.entrySet()) {
            builder.addNode(entry.getValue());
        }
        
        for (Map.Entry<String, Channel> entry : channels.entrySet()) {
            builder.addChannel(entry.getKey(), entry.getValue());
        }
        
        if (checkpointer != null) {
            builder.setCheckpointer(checkpointer);
        }
        
        Pregel pregel = builder.build();
        
        // Return the compiled graph
        return new CompiledStateGraph<>(pregel, stateValidator.getSchemaType());
    }
    
    /**
     * Create a PregelExecutable for a node.
     *
     * @param action Node action
     * @return PregelExecutable
     */
    @SuppressWarnings("unchecked")
    private PregelExecutable createNodeExecutable(NodeAction<S> action) {
        return (inputs, context) -> {
            Map<String, Object> result = new HashMap<>();
            
            // Get the current state
            Object stateObj = inputs.get("state");
            S state = (S) stateObj;
            
            // Execute the node action
            Map<String, Object> updates = action.execute(state);
            
            // Create updated state by merging updates
            Map<String, Object> newState;
            if (state instanceof Map) {
                // Handle Map state
                @SuppressWarnings("unchecked")
                Map<String, Object> stateMap = new HashMap<>((Map<String, Object>) state);
                stateMap.putAll(updates);
                newState = stateMap;
            } else {
                // Handle record state (create a copy with updates)
                newState = createUpdatedState(state, updates);
            }
            
            // Set the updated state
            result.put("state", newState);
            
            return result;
        };
    }
    
    /**
     * Create a PregelExecutable for the entry node.
     *
     * @return PregelExecutable
     */
    private PregelExecutable createEntryExecutable() {
        return (inputs, context) -> {
            Map<String, Object> result = new HashMap<>();
            
            // Get the input
            Object input = inputs.get("input");
            
            // Set the next node
            if (conditionalEntryPoint != null) {
                // Use conditional entry point
                @SuppressWarnings("unchecked")
                String nextNode = conditionalEntryPoint.route((S) input);
                result.put("next", nextNode);
            } else {
                // Use static entry point
                result.put("next", entryPoint);
            }
            
            // Set the initial state
            result.put("state", input);
            
            return result;
        };
    }
    
    /**
     * Create a PregelExecutable for the router node.
     *
     * @return PregelExecutable
     */
    private PregelExecutable createRouterExecutable() {
        return (inputs, context) -> {
            // Get the next node
            String nextNode = (String) inputs.get("next");
            
            // Check if we're done
            if (END.equals(nextNode) || (nextNode != null && finishPoints.contains(nextNode))) {
                // Signal completion
                return Collections.emptyMap();
            }
            
            // Check for conditional routing
            if (conditionalEdges.containsKey(nextNode)) {
                // Get the current state
                @SuppressWarnings("unchecked")
                S state = (S) context.get("state");
                
                // Get the next node from the condition
                EdgeCondition<S> condition = conditionalEdges.get(nextNode);
                String routedNode = condition.route(state);
                
                // Update the next node
                Map<String, Object> result = new HashMap<>();
                result.put("next", routedNode);
                return result;
            }
            
            // Check for static routing
            if (edges.containsKey(nextNode) && !edges.get(nextNode).isEmpty()) {
                // Get the first edge (assuming single edge for now)
                String routedNode = edges.get(nextNode).iterator().next();
                
                // Update the next node
                Map<String, Object> result = new HashMap<>();
                result.put("next", routedNode);
                return result;
            }
            
            // No routing found, signal completion
            return Collections.emptyMap();
        };
    }
    
    /**
     * Create an updated state by applying updates to a record.
     *
     * @param state Original state
     * @param updates Updates to apply
     * @return Updated state
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> createUpdatedState(S state, Map<String, Object> updates) {
        // Convert the record to a Map
        Map<String, Object> stateMap = new HashMap<>();
        
        for (java.lang.reflect.RecordComponent component : state.getClass().getRecordComponents()) {
            try {
                String name = component.getName();
                Object value = component.getAccessor().invoke(state);
                stateMap.put(name, value);
            } catch (Exception e) {
                throw new RuntimeException("Error accessing record component", e);
            }
        }
        
        // Apply updates
        stateMap.putAll(updates);
        
        return stateMap;
    }
}
```

## `CompiledStateGraph` Class

The executable result of compiling a StateGraph.

```java
package com.langgraph.graph;

import com.langgraph.pregel.PregelProtocol;
import com.langgraph.pregel.StreamMode;

import java.util.*;

/**
 * Executable result of compiling a StateGraph.
 *
 * @param <S> State type
 */
public class CompiledStateGraph<S> {
    private final PregelProtocol pregel;
    private final Class<?> stateType;
    
    /**
     * Create a CompiledStateGraph.
     *
     * @param pregel Pregel instance
     * @param stateType State type
     */
    public CompiledStateGraph(PregelProtocol pregel, Class<?> stateType) {
        this.pregel = pregel;
        this.stateType = stateType;
    }
    
    /**
     * Invoke the graph with an input state.
     *
     * @param input Initial state
     * @return Final state
     */
    @SuppressWarnings("unchecked")
    public S invoke(S input) {
        return invoke(input, null);
    }
    
    /**
     * Invoke the graph with an input state and configuration.
     *
     * @param input Initial state
     * @param config Configuration
     * @return Final state
     */
    @SuppressWarnings("unchecked")
    public S invoke(S input, Map<String, Object> config) {
        // Validate input
        if (input != null && !stateType.isInstance(input)) {
            throw new IllegalArgumentException(
                    "Input must be an instance of " + stateType.getName() + 
                    " or null");
        }
        
        // Create input map
        Map<String, Object> inputMap = new HashMap<>();
        inputMap.put("input", input);
        
        // Invoke the graph
        Object result = pregel.invoke(inputMap, config);
        
        // Extract the final state
        if (result instanceof Map) {
            Map<String, Object> resultMap = (Map<String, Object>) result;
            Object stateObj = resultMap.get("state");
            
            if (stateObj == null) {
                return null;
            }
            
            if (stateType.isInstance(stateObj)) {
                return (S) stateObj;
            } else if (stateObj instanceof Map) {
                // Convert Map to record (state type)
                return convertMapToState((Map<String, Object>) stateObj);
            }
        }
        
        return null;
    }
    
    /**
     * Stream the execution of the graph.
     *
     * @param input Initial state
     * @return Iterator of state updates
     */
    public Iterator<S> stream(S input) {
        return stream(input, null, StreamMode.VALUES);
    }
    
    /**
     * Stream the execution of the graph with configuration and mode.
     *
     * @param input Initial state
     * @param config Configuration
     * @param streamMode Stream mode
     * @return Iterator of state updates
     */
    @SuppressWarnings("unchecked")
    public Iterator<S> stream(S input, Map<String, Object> config, StreamMode streamMode) {
        // Validate input
        if (input != null && !stateType.isInstance(input)) {
            throw new IllegalArgumentException(
                    "Input must be an instance of " + stateType.getName() + 
                    " or null");
        }
        
        // Create input map
        Map<String, Object> inputMap = new HashMap<>();
        inputMap.put("input", input);
        
        // Stream the execution
        Iterator<Object> results = pregel.stream(inputMap, config, streamMode);
        
        // Convert results to state objects
        return new Iterator<S>() {
            @Override
            public boolean hasNext() {
                return results.hasNext();
            }
            
            @Override
            public S next() {
                Object result = results.next();
                
                if (result instanceof Map) {
                    Map<String, Object> resultMap = (Map<String, Object>) result;
                    Object stateObj = resultMap.get("state");
                    
                    if (stateObj == null) {
                        return null;
                    }
                    
                    if (stateType.isInstance(stateObj)) {
                        return (S) stateObj;
                    } else if (stateObj instanceof Map) {
                        // Convert Map to record (state type)
                        return convertMapToState((Map<String, Object>) stateObj);
                    }
                }
                
                return null;
            }
        };
    }
    
    /**
     * Get the current state for a thread.
     *
     * @param threadId Thread ID
     * @return Current state
     */
    @SuppressWarnings("unchecked")
    public S getState(String threadId) {
        Object state = pregel.getState(threadId);
        
        if (state instanceof Map) {
            Map<String, Object> stateMap = (Map<String, Object>) state;
            Object stateObj = stateMap.get("state");
            
            if (stateObj == null) {
                return null;
            }
            
            if (stateType.isInstance(stateObj)) {
                return (S) stateObj;
            } else if (stateObj instanceof Map) {
                // Convert Map to record (state type)
                return convertMapToState((Map<String, Object>) stateObj);
            }
        }
        
        return null;
    }
    
    /**
     * Update the state for a thread.
     *
     * @param threadId Thread ID
     * @param state New state
     */
    public void updateState(String threadId, S state) {
        // Validate state
        if (state != null && !stateType.isInstance(state)) {
            throw new IllegalArgumentException(
                    "State must be an instance of " + stateType.getName() + 
                    " or null");
        }
        
        // Create state map
        Map<String, Object> stateMap = new HashMap<>();
        stateMap.put("state", state);
        
        // Update the state
        pregel.updateState(threadId, stateMap);
    }
    
    /**
     * Get the state history for a thread.
     *
     * @param threadId Thread ID
     * @return List of state snapshots
     */
    @SuppressWarnings("unchecked")
    public List<S> getStateHistory(String threadId) {
        List<Object> history = pregel.getStateHistory(threadId);
        List<S> result = new ArrayList<>();
        
        for (Object snapshot : history) {
            if (snapshot instanceof Map) {
                Map<String, Object> stateMap = (Map<String, Object>) snapshot;
                Object stateObj = stateMap.get("state");
                
                if (stateObj == null) {
                    result.add(null);
                } else if (stateType.isInstance(stateObj)) {
                    result.add((S) stateObj);
                } else if (stateObj instanceof Map) {
                    // Convert Map to record (state type)
                    result.add(convertMapToState((Map<String, Object>) stateObj));
                }
            }
        }
        
        return result;
    }
    
    /**
     * Convert a Map to a state object.
     *
     * @param stateMap Map of state values
     * @return State object
     */
    @SuppressWarnings("unchecked")
    private S convertMapToState(Map<String, Object> stateMap) {
        if (stateType.isRecord()) {
            try {
                // Get the record components
                java.lang.reflect.RecordComponent[] components = stateType.getRecordComponents();
                
                // Create the constructor parameters
                Object[] params = new Object[components.length];
                
                for (int i = 0; i < components.length; i++) {
                    java.lang.reflect.RecordComponent component = components[i];
                    String name = component.getName();
                    params[i] = stateMap.get(name);
                }
                
                // Get the canonical constructor
                java.lang.reflect.Constructor<?> constructor = stateType.getDeclaredConstructor(
                        Arrays.stream(components)
                                .map(java.lang.reflect.RecordComponent::getType)
                                .toArray(Class[]::new)
                );
                
                // Create a new record instance
                return (S) constructor.newInstance(params);
            } catch (Exception e) {
                throw new RuntimeException("Error creating record instance", e);
            }
        }
        
        // Fallback: return the map as is
        return (S) stateMap;
    }
}
```

## Example Usage

```java
package com.langgraph.examples;

import com.langgraph.graph.StateGraph;
import com.langgraph.graph.CompiledStateGraph;
import com.langgraph.checkpoint.memory.MemoryCheckpointSaver;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.langgraph.graph.GraphConstants.END;
import static com.langgraph.graph.GraphConstants.START;

/**
 * Example of using StateGraph with a Record state.
 */
public class StateGraphExample {
    /**
     * State schema as a Java Record.
     */
    public record CounterState(int count, String message) {}
    
    public static void main(String[] args) {
        // Create a graph with our schema
        StateGraph<CounterState> graph = new StateGraph<>(CounterState.class);
        
        // Add nodes
        graph.addNode("increment", state -> {
            Map<String, Object> updates = new HashMap<>();
            updates.put("count", state.count() + 1);
            return updates;
        });
        
        graph.addNode("check", state -> {
            // No state changes, just routing
            return Map.of();
        });
        
        graph.addNode("finish", state -> {
            Map<String, Object> updates = new HashMap<>();
            updates.put("message", "Finished with count " + state.count());
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
        
        graph.addEdge("finish", END);
        
        // Create a memory checkpointer
        MemoryCheckpointSaver checkpointer = new MemoryCheckpointSaver();
        
        // Compile the graph
        CompiledStateGraph<CounterState> compiled = graph.compile(checkpointer);
        
        // Create initial state
        CounterState initialState = new CounterState(0, "");
        
        // Invoke the graph
        CounterState result = compiled.invoke(initialState);
        
        // Print the result
        System.out.println("Result: " + result);
        
        // Get state history
        List<CounterState> history = compiled.getStateHistory("default");
        
        // Print history
        System.out.println("History:");
        for (CounterState state : history) {
            System.out.println("  " + state);
        }
    }
}
```

### `MemoryCheckpointSaver` Implementation 

```java
package com.langgraph.checkpoint.memory;

import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.checkpoint.base.ID;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory implementation of a checkpoint saver.
 */
public class MemoryCheckpointSaver implements BaseCheckpointSaver {
    private final Map<String, Map<String, Object>> checkpoints = new ConcurrentHashMap<>();
    private final Map<String, List<String>> threadCheckpoints = new ConcurrentHashMap<>();
    
    @Override
    public String checkpoint(String threadId, Map<String, Object> channelValues) {
        String checkpointId = ID.checkpointId(threadId);
        
        // Store the checkpoint
        checkpoints.put(checkpointId, new HashMap<>(channelValues));
        
        // Add to thread's checkpoints
        threadCheckpoints.computeIfAbsent(threadId, k -> new ArrayList<>()).add(checkpointId);
        
        return checkpointId;
    }
    
    @Override
    public Optional<Map<String, Object>> getValues(String checkpointId) {
        Map<String, Object> values = checkpoints.get(checkpointId);
        return Optional.ofNullable(values).map(HashMap::new);
    }
    
    @Override
    public List<String> list(String threadId) {
        List<String> result = threadCheckpoints.get(threadId);
        return result != null ? new ArrayList<>(result) : Collections.emptyList();
    }
    
    @Override
    public Optional<String> latest(String threadId) {
        List<String> checkpoints = threadCheckpoints.get(threadId);
        
        if (checkpoints == null || checkpoints.isEmpty()) {
            return Optional.empty();
        }
        
        return Optional.of(checkpoints.get(checkpoints.size() - 1));
    }
    
    @Override
    public void delete(String checkpointId) {
        // Remove the checkpoint
        Map<String, Object> removed = checkpoints.remove(checkpointId);
        
        if (removed != null) {
            // Remove from thread's checkpoints
            for (List<String> threadCheckpointList : threadCheckpoints.values()) {
                threadCheckpointList.remove(checkpointId);
            }
        }
    }
    
    @Override
    public void clear(String threadId) {
        List<String> checkpointIds = threadCheckpoints.remove(threadId);
        
        if (checkpointIds != null) {
            // Remove all checkpoints for this thread
            for (String checkpointId : checkpointIds) {
                checkpoints.remove(checkpointId);
            }
        }
    }
}
```