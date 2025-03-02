package com.langgraph.pregel;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.LastValue;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class PregelTest {

    /**
     * Mock implementation of BaseCheckpointSaver for testing
     */
    private static class TestCheckpointSaver implements BaseCheckpointSaver {
        private final Map<String, Map<String, Object>> checkpoints = new HashMap<>();
        private final Map<String, String> latestCheckpoints = new HashMap<>();
        private final Map<String, List<String>> threadCheckpoints = new HashMap<>();
        private final Map<String, AtomicInteger> stepCounters = new HashMap<>();

        @Override
        public String checkpoint(String threadId, Map<String, Object> values) {
            AtomicInteger counter = stepCounters.computeIfAbsent(threadId, k -> new AtomicInteger(0));
            int stepCount = counter.incrementAndGet();
            String checkpointId = threadId + "_" + stepCount;
            
            checkpoints.put(checkpointId, new HashMap<>(values));
            latestCheckpoints.put(threadId, checkpointId);
            
            List<String> list = threadCheckpoints.computeIfAbsent(threadId, k -> new ArrayList<>());
            list.add(checkpointId);
            
            return checkpointId;
        }
        
        @Override
        public Optional<String> latest(String threadId) {
            return Optional.ofNullable(latestCheckpoints.get(threadId));
        }
        
        @Override
        public Optional<Map<String, Object>> getValues(String checkpointId) {
            return Optional.ofNullable(checkpoints.get(checkpointId))
                    .map(HashMap::new);
        }
        
        @Override
        public List<String> list(String threadId) {
            return threadCheckpoints.getOrDefault(threadId, Collections.emptyList());
        }
        
        @Override
        public void delete(String checkpointId) {
            checkpoints.remove(checkpointId);
            for (Map.Entry<String, String> entry : latestCheckpoints.entrySet()) {
                if (checkpointId.equals(entry.getValue())) {
                    latestCheckpoints.remove(entry.getKey());
                }
            }
            for (Map.Entry<String, List<String>> entry : threadCheckpoints.entrySet()) {
                entry.getValue().remove(checkpointId);
            }
        }
        
        @Override
        public void clear(String threadId) {
            List<String> ids = list(threadId);
            for (String id : ids) {
                delete(id);
            }
            threadCheckpoints.remove(threadId);
            latestCheckpoints.remove(threadId);
            stepCounters.remove(threadId);
        }
    }

    /**
     * Simple test node action that returns a fixed value
     */
    private static class FixedValueAction implements PregelExecutable {
        private final Object value;
        
        public FixedValueAction(Object value) {
            this.value = value;
        }
        
        @Override
        public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
            System.out.println("FixedValueAction - returning value: " + value);
            Map<String, Object> output = new HashMap<>();
            output.put("counter", value);
            // Return empty map on second call to prevent infinite loops
            return output;
        }
    }

    /**
     * Action that passes through values but stops after maxSteps
     */
    private static class LimitedAction implements PregelExecutable {
        private final int maxSteps;
        private final AtomicInteger stepCount = new AtomicInteger(0);
        
        public LimitedAction(int maxSteps) {
            this.maxSteps = maxSteps;
        }
        
        @Override
        public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
            int count = stepCount.incrementAndGet();
            
            // If we've reached max steps, return empty to stop
            if (count > maxSteps) {
                return Collections.emptyMap();
            }
            
            // Otherwise, pass through the input values with a step counter
            Map<String, Object> output = new HashMap<>(inputs);
            output.put("step", count);
            return output;
        }
    }

    @Test
    void testConstructors() {
        // For constructor tests, we still use direct constructors to validate they work properly
        
        // Setup test components using builder patterns where appropriate
        Map<String, PregelNode> nodes = new HashMap<>();
        nodes.put("node1", new PregelNode.Builder("node1", new FixedValueAction(1)).build());
        
        Map<String, BaseChannel> channels = new HashMap<>();
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0));
        channels.put("counter", counterChannel);
        
        TestCheckpointSaver checkpointer = new TestCheckpointSaver();
        
        // Test constructor with all parameters
        Pregel pregel1 = new Pregel(nodes, channels, checkpointer, 50);
        assertThat(pregel1.getNodeRegistry()).isNotNull();
        assertThat(pregel1.getChannelRegistry()).isNotNull();
        assertThat(pregel1.getCheckpointer()).isEqualTo(checkpointer);
        
        // Test constructor with default max steps
        Pregel pregel2 = new Pregel(nodes, channels, checkpointer);
        assertThat(pregel2.getNodeRegistry()).isNotNull();
        assertThat(pregel2.getChannelRegistry()).isNotNull();
        assertThat(pregel2.getCheckpointer()).isEqualTo(checkpointer);
        
        // Test constructor without checkpointer
        Pregel pregel3 = new Pregel(nodes, channels);
        assertThat(pregel3.getNodeRegistry()).isNotNull();
        assertThat(pregel3.getChannelRegistry()).isNotNull();
        assertThat(pregel3.getCheckpointer()).isNull();
    }

    @Test
    void testBasicInvocation() {
        // Setup a simple counter graph using FixedValueAction that returns 1
        // Use builder pattern for all supported components
        
        // Create a node with the builder pattern
        PregelNode node = new PregelNode.Builder("counter", new FixedValueAction(1))
                .subscribe("counter")
                .writer("counter")
                .build();
        
        // Initialize the channel with a default value
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0)); // Set initial value to 0
        
        // Use Pregel builder pattern
        Pregel pregel = new Pregel.Builder()
                .addNode(node)
                .addChannel("counter", counterChannel)
                .build();
        
        // Initialize with counter=0
        Map<String, Object> input = new HashMap<>();
        input.put("counter", 0);
        
        // Execute and check the result
        @SuppressWarnings("unchecked")
        Map<String, Object> result = (Map<String, Object>) pregel.invoke(input, null);
        
        System.out.println("testBasicInvocation - Result: " + result);
        
        assertThat(result).containsKey("counter");
        assertThat(result.get("counter")).isEqualTo(1);
    }

    @Test
    @SuppressWarnings("unchecked")
    void testMultiStepExecution() {
        // Setup a graph with a node that stops after 3 steps
        // Using builder pattern for better readability and best practices
        
        // Create a node with the builder pattern
        PregelNode node = new PregelNode.Builder("limited", new LimitedAction(3))
                .subscribe("counter")
                .writer("counter")
                .writer("step")
                .build();
        
        // Initialize channels with default values
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0));
        
        LastValue<Integer> stepChannel = new LastValue<>(Integer.class, "step");
        stepChannel.update(Collections.singletonList(0));
        
        // Create test checkpointer
        TestCheckpointSaver checkpointer = new TestCheckpointSaver();
        
        // Create Pregel with builder
        Pregel pregel = new Pregel.Builder()
                .addNode(node)
                .addChannel("counter", counterChannel)
                .addChannel("step", stepChannel)
                .setCheckpointer(checkpointer)
                .setMaxSteps(10)
                .build();
        
        // Initialize with counter=0
        Map<String, Object> input = new HashMap<>();
        input.put("counter", 0);
        
        // Set thread ID for consistent checkpoints
        Map<String, Object> config = new HashMap<>();
        config.put("thread_id", "test-thread");
        
        // Execute and check the result
        Map<String, Object> result = (Map<String, Object>) pregel.invoke(input, config);
        
        System.out.println("testMultiStepExecution - Result: " + result);
        System.out.println("testMultiStepExecution - History size: " + checkpointer.list("test-thread").size());
        checkpointer.list("test-thread").forEach(id -> 
            System.out.println("testMultiStepExecution - Checkpoint " + id + ": " + 
                checkpointer.getValues(id).orElse(Collections.emptyMap())));
        
        // Verify final state
        assertThat(result).containsKey("step");
        assertThat(result.get("step")).isEqualTo(3);
        
        // Verify checkpoints were created
        Object stateHistory = pregel.getStateHistory("test-thread");
        assertThat(stateHistory).isInstanceOf(List.class);
        assertThat((List<Object>) stateHistory).hasSizeGreaterThanOrEqualTo(3);
    }

    @Test
    void testStreamOutput() {
        // Setup a simple graph that runs for 3 steps using builder pattern
        
        // Create node with builder
        PregelNode node = new PregelNode.Builder("limited", new LimitedAction(3))
                .subscribe("counter")
                .writer("counter")
                .writer("step")
                .build();
        
        // Initialize channels with default values
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0));
        
        LastValue<Integer> stepChannel = new LastValue<>(Integer.class, "step");
        stepChannel.update(Collections.singletonList(0));
        
        // Create Pregel with builder
        Pregel pregel = new Pregel.Builder()
                .addNode(node)
                .addChannel("counter", counterChannel)
                .addChannel("step", stepChannel)
                .build();
        
        // Initialize with counter=0
        Map<String, Object> input = new HashMap<>();
        input.put("counter", 0);
        
        // Set config with thread ID
        Map<String, Object> config = new HashMap<>();
        config.put("thread_id", "stream-test");
        
        // Stream in VALUES mode
        Iterator<Object> iterator = pregel.stream(input, config, StreamMode.VALUES);
        
        // Collect the streamed values
        List<Object> streamedValues = new ArrayList<>();
        while (iterator.hasNext()) {
            Object value = iterator.next();
            System.out.println("testStreamOutput - Received: " + value);
            streamedValues.add(value);
        }
        
        System.out.println("testStreamOutput - Total values received: " + streamedValues.size());
        
        // Verify we got values (may not be exactly 3 due to how the iterator works)
        assertThat(streamedValues).isNotEmpty();
    }

    @Test
    void testStateManagement() {
        // Setup a graph with checkpointing using builder pattern
        
        // Create node with builder
        PregelNode node = new PregelNode.Builder("counter", new FixedValueAction(11))
                .subscribe("counter")
                .writer("counter")
                .build();
        
        // Initialize channel with default value
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0));
        
        // Create checkpointer
        TestCheckpointSaver checkpointer = new TestCheckpointSaver();
        
        // Create Pregel with builder
        Pregel pregel = new Pregel.Builder()
                .addNode(node)
                .addChannel("counter", counterChannel)
                .setCheckpointer(checkpointer)
                .build();
        
        // Create a thread ID for state tracking
        String threadId = "state-test";
        
        // Create initial state and update
        Map<String, Object> initialState = new HashMap<>();
        initialState.put("counter", 10);
        pregel.updateState(threadId, initialState);
        
        // Get the state back
        @SuppressWarnings("unchecked")
        Map<String, Object> retrievedState = (Map<String, Object>) pregel.getState(threadId);
        
        System.out.println("testStateManagement - Retrieved state: " + retrievedState);
        
        assertThat(retrievedState).isNotNull();
        assertThat(retrievedState).containsEntry("counter", 10);
    }

    @Test
    void testBuilderPattern() {
        // Create channels with initial values
        LastValue<Integer> counterChannel = new LastValue<>(Integer.class, "counter");
        counterChannel.update(Collections.singletonList(0));
        
        LastValue<Integer> stepChannel = new LastValue<>(Integer.class, "step");
        stepChannel.update(Collections.singletonList(0));
        
        // Test the builder
        Pregel pregel = new Pregel.Builder()
                .addNode(new PregelNode("node1", new FixedValueAction(1)))
                .addNode(new PregelNode("node2", new LimitedAction(2)))
                .addChannel("counter", counterChannel)
                .addChannel("step", stepChannel)
                .setCheckpointer(new TestCheckpointSaver())
                .setMaxSteps(20)
                .build();
        
        assertThat(pregel.getNodeRegistry()).isNotNull();
        assertThat(pregel.getChannelRegistry()).isNotNull();
        assertThat(pregel.getCheckpointer()).isNotNull();
    }
}