# Java Port Implementation Plan for LangGraph

This document outlines the plan for implementing a Java port of LangGraph, following the specifications defined in the other documents.

## Project Structure

```
langgraph-java/
├── build.gradle
├── settings.gradle
├── README.md
├── langgraph-checkpoint/           # Base persistence interfaces (equiv. to libs/checkpoint)
│   └── src/
│       ├── main/java/
│       │   └── com/langgraph/checkpoint/
│       │       ├── base/           # Base interfaces and utilities
│       │       └── serde/          # Serialization
│       └── test/java/
├── langgraph-core/                 # Main library (equiv. to libs/langgraph)
│   └── src/
│       ├── main/java/
│       │   └── com/langgraph/
│       │       ├── channels/       # Channel implementations
│       │       ├── graph/          # StateGraph API
│       │       └── pregel/         # Pregel execution engine
│       └── test/java/
├── langgraph-examples/             # Example applications
│   └── src/
│       └── main/java/
│           └── com/langgraph/examples/
└── langgraph-checkpoint-postgres/  # Postgres implementation (future)
```

## Development Phases

The implementation will follow a bottom-up approach, starting with the lowest-level components and building up to the high-level APIs.

### Phase 1: Foundation (langgraph-checkpoint)

1. **Set up project structure**

   - Initialize Gradle project
   - Configure dependencies
   - Set up testing framework (JUnit 5)

2. **Implement ID utilities**

   - `ID.java` - For deterministic ID generation
   - Test with different inputs

3. **Implement serialization framework**

   - `Serializer` interface
   - `ReflectionSerializer` interface
   - `TypeSerializer` and `TypeDeserializer` interfaces
   - `MsgPackSerializer` implementation
   - Comprehensive tests for serialization/deserialization

4. **Implement checkpoint interfaces**
   - `BaseCheckpointSaver` interface
   - `AsyncBaseCheckpointSaver` interface
   - `MemoryCheckpointSaver` implementation
   - Tests for checkpoint operations

### Phase 2: State Management (langgraph-core)

1. **Implement channel system**

   - `Channel` interface
   - Basic channel implementations:
     - `LastValue`
     - `AnyValue`
     - `EphemeralValue`
     - `UntrackedValue`
   - Advanced channel implementations:
     - `Topic`
     - `BinaryOperatorAggregate`
     - `NamedBarrierValue`
   - Tests for each channel type

2. **Implement schema validation**
   - `SchemaValidator` interface
   - `RecordSchemaValidator` implementation
   - Tests for schema validation

### Phase 3: Execution Engine (langgraph-core)

1. **Implement core Pregel interfaces**

   - `PregelProtocol` interface
   - `StreamMode` enum
   - `PregelExecutable` interface
   - `PregelNode` class
   - `PregelTask` classes
   - Tests for interfaces

2. **Implement execution system**
   - `RetryPolicy` interface
   - `Checkpoint` class
   - `Pregel` implementation
   - Tests for execution

### Phase 4: High-Level API (langgraph-core)

1. **Implement StateGraph**

   - `GraphConstants` constants
   - `NodeAction` and `EdgeCondition` interfaces
   - `StateGraph` class
   - `CompiledStateGraph` class
   - Tests for graph construction and execution

2. **Build examples**
   - Counter example
   - Conversation example
   - Multi-step workflow example

### Phase 5: Extensions (future)

1. **Implement Postgres adaptors**
   - PostgresCheckpointSaver
   - Tests for Postgres integration

## Testing Strategy

Each component will be implemented following Test-Driven Development (TDD):

1. Write test cases based on Python implementation
2. Implement interface to satisfy tests
3. Implement concrete class(es)
4. Run tests and refine implementation
5. Document completed component

### Key Test Areas

- **Serialization**: Test with various object types, nested structures, and cycles
- **Channels**: Test all channel behaviors and edge cases
- **Pregel**: Test deterministic execution, error handling, and checkpoint integration
- **StateGraph**: Test graph construction, validation, and execution

## Implementation Notes

### Java-Specific Adaptations

1. **Records for state schemas**

   - Java Records (Java 14+) as TypedDict/Pydantic alternative
   - Reflection-based validation and conversion

2. **CompletableFuture for async**

   - Async interfaces use CompletableFuture
   - Parallel execution with work-stealing pools

3. **Generics for type safety**

   - Extensive use of generics for type safety
   - Runtime type checking for dynamic aspects

4. **Builder pattern**
   - Builder pattern for complex object construction
   - Fluent interfaces for API usability

### Performance Considerations

- Thread-safe implementations using concurrent collections
- Atomic operations for parallel safety
- Minimizing object creation and copying
- Efficient serialization with MessagePack

## Initial Implementation Tasks

1. Set up project structure and build system
2. Implement ID utilities in checkpoint module
3. Implement MsgPackSerializer
4. Implement MemoryCheckpointSaver
5. Write comprehensive tests for foundation layer

## Timeline

1. **Phase 1: Foundation** - 2 weeks
2. **Phase 2: State Management** - 2 weeks
3. **Phase 3: Execution Engine** - 3 weeks
4. **Phase 4: High-Level API** - 2 weeks
5. **Integration and Examples** - 1 week

Total estimated time: ~10 weeks for core functionality

## Dependencies

- **MessagePack**: `org.msgpack:msgpack-core:0.9.3`
- **JUnit 5**: `org.junit.jupiter:junit-jupiter:5.8.2`
- **Mockito**: `org.mockito:mockito-core:4.5.1`
- **AssertJ**: `org.assertj:assertj-core:3.22.0`

## Phase 3 in detail:

This implementation plan is organized to match the structure of the Python implementation while prioritizing testability at each step. Don't forget to use Java patterns and idioms where appropriate.

1. Core Data Structures & Interfaces (Week 1)

Day 1-2: Base Interfaces and Types

1. StreamMode Enum
   - Implementation: Define values (VALUES, UPDATES, DEBUG)
   - Test: Verify serialization and string representation
2. PregelExecutable Interface
   - Implementation: Define execute method
   - Test: Create mock implementations for testing
3. RetryPolicy Interface & Implementations
   - Implementation: RetryPolicy interface with factory methods
   - Test: Verify retry decision logic

Day 3-4: Task Management

1. PregelTask Class
   - Implementation: Node name, trigger, retry policy
   - Test: Constructor, getters, equals, hashCode
2. PregelExecutableTask Class
   - Implementation: Task with inputs and context
   - Test: Construction, input/context immutability

Day 5: Core Protocol

1. PregelProtocol Interface

   - Implementation: Define methods from spec
   - Test: Mock implementation for testing

2. Node & Channel Integration (Week 2)

Day 1-2: Node Implementation

1. PregelNode Class
   - Implementation: Full implementation with accessors
   - Test: Subscription, trigger, write permissions
2. NodeRegistry
   - Implementation: Managing collections of nodes
   - Test: Registration, lookup, validation

Day 3-5: Message Handling

1. Checkpoint Class

   - Implementation: State snapshot storage
   - Test: Capture and restore state

2. ChannelRegistry

   - Implementation: Managing channel collections
   - Test: Registration, lookup, validation

3. Core Execution Components (Week 3)

Day 1-2: Task Planning

1. PregelTaskPlanner
   - Implementation: Determine tasks to execute based on updates
   - Test: Task selection with different update patterns
2. TaskPrioritizer
   - Implementation: Order tasks for execution
   - Test: Priority ordering with different dependency patterns

Day 3-5: Task Execution

1. TaskExecutor
   - Implementation: Execute tasks with retry logic
   - Test: Successful execution, error handling, retry behavior
2. ExecutionContext

   - Implementation: Thread-local context for execution
   - Test: Context propagation, thread safety

3. Superstep Management (Week 4)

Day 1-2: Superstep Core

1. SuperstepManager
   - Implementation: Manage a single superstep execution
   - Test: Plan, execute, update phases with mock nodes
2. UpdateCollector
   - Implementation: Collect and apply channel updates
   - Test: Update ordering, conflict resolution

Day 3-5: Execution Loop

1. PregelLoop
   - Implementation: Core execution logic, step iteration
   - Test: Loop termination, state tracking
2. CheckpointManager

   - Implementation: Integration with checkpointing
   - Test: Checkpoint captures, restore behavior

3. Full Engine & Streaming (Week 5)

Day 1-3: Pregel Engine

1. Pregel Class
   - Implementation: Core engine with all components
   - Test: End-to-end execution, configuration
2. PregelBuilder
   - Implementation: Fluent builder interface
   - Test: Configuration options, validation

Day 4-5: Streaming Support

1. StreamOutput
   - Implementation: Format output for streaming
   - Test: Different stream modes
2. StreamController
   - Implementation: Manage streaming state
   - Test: Backpressure, cancellation

Implementation Strategy

1. Incremental Testing

Create test classes for each component that can be used in isolation:

```java
@Test
void testTaskPlanning() {
		// Create mock channels with updates
		Map<String, Channel> channels = createMockChannels(
				Map.of("input", true, "other", false));

		// Create nodes with subscriptions
		Set<PregelNode> nodes = createTestNodes();

		// Create planner
		PregelTaskPlanner planner = new PregelTaskPlanner(nodes);

		// Test planning logic
		List<PregelTask> tasks = planner.plan(channels);

		// Verify correct tasks selected
		assertThat(tasks).hasSize(1);
		assertThat(tasks.get(0).getNode()).isEqualTo("processor");
}
```

2. Test Each Component in Isolation

For each component, test:

- Normal operation
- Edge cases
- Error conditions
- Integration with dependencies

```java
@Test
void testTaskExecution() {
		// Create mock executor
		TaskExecutor executor = new TaskExecutor();

		// Create task with expected inputs
		PregelExecutableTask task = createTestTask();

		// Execute and capture results
		Map<String, Object> result = executor.execute(task);

		// Verify results
		assertThat(result)
				.containsKey("output")
				.containsEntry("output", "processed");
}
```

3. Incremental Integration

1. Start with simplest components: PregelTask, StreamMode, etc.
1. Build TaskPlanner with mocked nodes
1. Create TaskExecutor with mocked actions
1. Integrate into SuperstepManager
1. Combine in PregelLoop
1. Build complete Pregel engine

```java
// First, test task planner alone
@Test
void testTaskPlannerInIsolation() {
		PregelTaskPlanner planner = new PregelTaskPlanner(mockNodes);
		List<PregelTask> tasks = planner.plan(updatedChannels);
		// Verify tasks
}

// Then, test executor alone
@Test
void testTaskExecutorInIsolation() {
		TaskExecutor executor = new TaskExecutor();
		Map<String, Object> result = executor.execute(mockTask);
		// Verify result
}

// Finally, test them together in SuperstepManager
@Test
void testSuperstepIntegration() {
		SuperstepManager manager = new SuperstepManager(
				planner,
				executor,
				channels
		);
		SuperstepResult result = manager.executeStep();
		// Verify complete superstep behavior
}
```

4. Use Real Components When Possible

1. Use real Channel implementations from previous phase
1. Create simple test PregelExecutables
1. Build test workflows of increasing complexity

```java
@Test
void testSimpleWorkflow() {
		// Create real channels
		Map<String, Channel> channels = new HashMap<>();
		channels.put("input", new LastValue<>(String.class));
		channels.put("output", new LastValue<>(String.class));

		// Create real nodes
		Map<String, PregelNode> nodes = new HashMap<>();
		nodes.put("processor", new PregelNode(
				"processor",
				(inputs, context) -> {
						String input = (String) inputs.get("input");
						return Map.of("output", input.toUpperCase());
				},
				Set.of("input"),
				null,
				Set.of("output"),
				null
		));

		// Create real Pregel instance
		Pregel pregel = new Pregel(nodes, channels, null);

		// Run and verify
		Object result = pregel.invoke(Map.of("input", "hello"), null);

		// Verify complete execution
		@SuppressWarnings("unchecked")
		Map<String, Object> resultMap = (Map<String, Object>) result;
		assertThat(resultMap).containsEntry("output", "HELLO");
}
```

File Organization

Based on the Python structure, here's how the Java implementation will be organized:

com.langgraph.pregel/
├── PregelProtocol.java # Core interface
├── StreamMode.java # Enum for streaming options
├── PregelExecutable.java # Interface for node functions
├── PregelNode.java # Node definition
├── task/
│ ├── PregelTask.java # Task representation
│ ├── PregelExecutableTask.java # Task with inputs
│ ├── TaskPlanner.java # Task planning logic
│ └── TaskExecutor.java # Task execution
├── state/
│ ├── Checkpoint.java # State checkpoint
│ ├── ChannelRegistry.java # Channel management
│ └── NodeRegistry.java # Node management
├── execute/
│ ├── SuperstepManager.java # Single superstep execution
│ ├── PregelLoop.java # Main execution loop
│ ├── ExecutionContext.java # Context for execution
│ └── UpdateCollector.java # Collect updates
├── stream/
│ ├── StreamController.java # Manage streaming
│ └── StreamOutput.java # Format output
├── retry/
│ ├── RetryPolicy.java # Retry interface
│ └── RetryPolicies.java # Standard policies
└── Pregel.java # Main implementation

Testing Step-by-Step

The testing strategy follows a specific progression:

1. Unit Testing: Test each component in isolation
2. Component Testing: Test related components together
3. Integration Testing: Test main subsystems
4. System Testing: Test complete workflows

Example Test Progression for TaskPlanner:

1. Unit Test: Mock everything

```java
@Test
void testPlannerWithMocks() {
		Set<String> updatedChannels = Set.of("input");
		Map<String, PregelNode> mockNodes = createMockNodes();

		TaskPlanner planner = new TaskPlanner(mockNodes);
		List<PregelTask> tasks = planner.plan(updatedChannels);

		// Test with various update patterns
}
```

2. Component Test: Use real nodes, mock channels

```java
@Test
void testPlannerWithRealNodes() {
		Set<String> updatedChannels = Set.of("input");
		Map<String, PregelNode> realNodes = createRealNodes();

		TaskPlanner planner = new TaskPlanner(realNodes);
		List<PregelTask> tasks = planner.plan(updatedChannels);

		// Verify with real node behavior
}
```

3. Integration Test: Use real nodes and channels

```java
@Test
void testPlannerIntegration() {
		Map<String, Channel> channels = createRealChannels();
		// Update channels
		channels.get("input").update("test");

		Map<String, PregelNode> nodes = createRealNodes();

		// Get updated channel names
		Set<String> updatedChannels = getUpdatedChannelNames(channels);

		TaskPlanner planner = new TaskPlanner(nodes);
		List<PregelTask> tasks = planner.plan(updatedChannels);

		// Verify end-to-end planning
}
```

4. System Test: Use in a full Pregel execution

```java
@Test
void testPlannerInFullSystem() {
		// Set up complete Pregel system
		Pregel pregel = createTestPregelSystem();

		// Execute a workflow that will trigger planning
		pregel.invoke(Map.of("input", "test"), null);

		// Verify entire execution via output
}
```

Sample Test Case Implementations

To illustrate the TDD approach, here are key test cases for early components:

1. PregelTask

```java
@Test
void testPregelTask() {
		// Basic construction
		PregelTask task = new PregelTask("node1", "trigger1", RetryPolicy.noRetry());

		assertThat(task.getNode()).isEqualTo("node1");
		assertThat(task.getTrigger()).isEqualTo("trigger1");
		assertThat(task.getRetryPolicy()).isNotNull();

		// Equality
		PregelTask sameTask = new PregelTask("node1", "trigger1", RetryPolicy.maxAttempts(3));
		PregelTask differentNode = new PregelTask("node2", "trigger1", RetryPolicy.noRetry());
		PregelTask differentTrigger = new PregelTask("node1", "trigger2", RetryPolicy.noRetry());

		assertThat(task).isEqualTo(sameTask);
		assertThat(task).isNotEqualTo(differentNode);
		assertThat(task).isNotEqualTo(differentTrigger);
}
```

2. PregelNode

```java
@Test
void testPregelNode() {
		// Create a simple action
		PregelExecutable action = (inputs, context) -> Map.of("output", "result");

		// Basic construction
		PregelNode node = new PregelNode(
				"processor",
				action,
				Set.of("input1", "input2"),
				"trigger1",
				Set.of("output1", "output2"),
				RetryPolicy.maxAttempts(3)
		);

		// Test properties
		assertThat(node.getName()).isEqualTo("processor");
		assertThat(node.getAction()).isSameAs(action);
		assertThat(node.getSubscribe()).containsExactlyInAnyOrder("input1", "input2");
		assertThat(node.getTrigger()).isEqualTo("trigger1");
		assertThat(node.getWriters()).containsExactlyInAnyOrder("output1", "output2");
		assertThat(node.getRetryPolicy()).isNotNull();

		// Test helper methods
		assertThat(node.subscribesTo("input1")).isTrue();
		assertThat(node.subscribesTo("input3")).isFalse();
		assertThat(node.hasTrigger("trigger1")).isTrue();
		assertThat(node.hasTrigger("trigger2")).isFalse();
		assertThat(node.canWriteTo("output1")).isTrue();
		assertThat(node.canWriteTo("output3")).isFalse();
}
```

3. TaskPlanner

```java
@Test
void testTaskPlanner() {
		// Create nodes
		PregelNode node1 = new PregelNode(
				"node1",
				(inputs, context) -> Map.of(),
				Set.of("channel1"),
				null,
				Set.of("output1"),
				null
		);

		PregelNode node2 = new PregelNode(
				"node2",
				(inputs, context) -> Map.of(),
				Set.of("channel2"),
				null,
				Set.of("output2"),
				null
		);

		PregelNode node3 = new PregelNode(
				"node3",
				(inputs, context) -> Map.of(),
				null,
				"trigger1",
				Set.of("output3"),
				null
		);

		Map<String, PregelNode> nodes = Map.of(
				"node1", node1,
				"node2", node2,
				"node3", node3
		);

		TaskPlanner planner = new TaskPlanner(nodes);

		// Test with different updated channels
		Set<String> update1 = Set.of("channel1");
		List<PregelTask> tasks1 = planner.plan(update1);
		assertThat(tasks1).hasSize(1);
		assertThat(tasks1.get(0).getNode()).isEqualTo("node1");

		Set<String> update2 = Set.of("channel1", "channel2");
		List<PregelTask> tasks2 = planner.plan(update2);
		assertThat(tasks2).hasSize(2);

		Set<String> update3 = Set.of("trigger1");
		List<PregelTask> tasks3 = planner.plan(update3);
		assertThat(tasks3).hasSize(1);
		assertThat(tasks3.get(0).getNode()).isEqualTo("node3");

		Set<String> update4 = Set.of("channel3");
		List<PregelTask> tasks4 = planner.plan(update4);
		assertThat(tasks4).isEmpty();
}
```
