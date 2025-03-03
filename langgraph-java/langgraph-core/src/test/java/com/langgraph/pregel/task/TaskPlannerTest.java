package com.langgraph.pregel.task;

import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;
import com.langgraph.pregel.retry.RetryPolicy;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TaskPlannerTest {
    
    private PregelNode<Object, Object> node1;
    private PregelNode<Object, Object> node2;
    private PregelNode<Object, Object> node3;
    private Map<String, PregelNode<?, ?>> nodes;
    private RetryPolicy testRetryPolicy;
    
    @BeforeEach
    void setUp() {
        // Create a simple executable for testing with proper generic type
        PregelExecutable<Object, Object> simpleExecutable = (inputs, context) -> Collections.emptyMap();
        
        // Create real nodes with proper generic types
        node1 = new PregelNode.Builder<Object, Object>("node1", simpleExecutable)
                .channels(Collections.singleton("channel1"))
                .build();
        
        node2 = new PregelNode.Builder<Object, Object>("node2", simpleExecutable)
                .channels(Arrays.asList("channel2", "channel3"))
                .build();
        
        testRetryPolicy = RetryPolicy.maxAttempts(3);
        
        node3 = new PregelNode.Builder<Object, Object>("node3", simpleExecutable)
                .triggerChannels("channel4")
                .retryPolicy(testRetryPolicy)
                .build();
        
        // Create nodes map with proper generic types
        nodes = new HashMap<>();
        nodes.put("node1", node1);
        nodes.put("node2", node2);
        nodes.put("node3", node3);
    }
    
    @Test
    void testConstructorWithNullNodes() {
        assertThatThrownBy(() -> new TaskPlanner(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null");
    }
    
    @Test
    void testPlanWithEmptyUpdatedChannels() {
        // Setup test data with input channel as trigger
        PregelExecutable<Object, Object> simpleExecutable = (inputs, context) -> Collections.emptyMap();
        
        // Create nodes with "input" as trigger
        PregelNode<Object, Object> inputNode = new PregelNode.Builder<Object, Object>("inputNode", simpleExecutable)
                .triggerChannels("input")
                .build();
                
        Map<String, PregelNode<?, ?>> nodesWithInputTrigger = new HashMap<>();
        nodesWithInputTrigger.put("inputNode", inputNode);
        
        // Create planner with nodes that have input trigger
        TaskPlanner planner = new TaskPlanner(nodesWithInputTrigger);
        
        // With Python compatibility, only nodes with input trigger should execute on first run
        List<PregelTask> tasks = planner.plan(Collections.emptyList());
        assertThat(tasks).hasSize(1);
        assertThat(tasks).extracting(PregelTask::getNode)
            .containsExactly("inputNode");
        
        // Also test with null updated channels
        tasks = planner.plan(null);
        assertThat(tasks).hasSize(1);
        assertThat(tasks).extracting(PregelTask::getNode)
            .containsExactly("inputNode");
            
        // Test with regular nodes (no input triggers) should return empty list
        TaskPlanner regularPlanner = new TaskPlanner(nodes);
        tasks = regularPlanner.plan(Collections.emptyList());
        assertThat(tasks).isEmpty();
    }
    
    @Test
    void testPlanWithSubscribedChannels() {
        TaskPlanner planner = new TaskPlanner(nodes);
        
        // Update channel1, should trigger node1
        List<PregelTask> tasks = planner.plan(Collections.singleton("channel1"));
        
        assertThat(tasks).hasSize(1);
        assertThat(tasks.get(0).getNode()).isEqualTo("node1");
        assertThat(tasks.get(0).getTrigger()).isNull();
        assertThat(tasks.get(0).getRetryPolicy()).isNull();
        
        // Update channel2 and channel3, should trigger node2
        tasks = planner.plan(Arrays.asList("channel2", "channel3"));
        
        assertThat(tasks).hasSize(1);
        assertThat(tasks.get(0).getNode()).isEqualTo("node2");
        
        // Update multiple channels, should trigger multiple nodes
        tasks = planner.plan(Arrays.asList("channel1", "channel2"));
        
        assertThat(tasks).hasSize(2);
        assertThat(tasks).extracting(PregelTask::getNode).containsExactlyInAnyOrder("node1", "node2");
    }
    
    @Test
    void testPlanWithTriggeredChannels() {
        TaskPlanner planner = new TaskPlanner(nodes);
        
        // Update channel4, should trigger node3 via its trigger
        List<PregelTask> tasks = planner.plan(Collections.singleton("channel4"));
        
        assertThat(tasks).hasSize(1);
        assertThat(tasks.get(0).getNode()).isEqualTo("node3");
        assertThat(tasks.get(0).getTrigger()).isEqualTo("channel4");
        assertThat(tasks.get(0).getRetryPolicy()).isNotNull();
    }
    
    @Test
    void testPlanAndPrioritizeCallsAllMethods() {
        // We'll create a custom planner that tracks method calls
        final boolean[] planCalled = {false};
        final boolean[] filterCalled = {false};
        final boolean[] prioritizeCalled = {false};
        
        // Create test tasks to verify they pass through all methods
        Set<String> updatedChannels = new HashSet<>(Arrays.asList("channel1", "channel4"));
        List<PregelTask> expectedTasks = Arrays.asList(
                new PregelTask("node1", null, null),
                new PregelTask("node3", "channel4", testRetryPolicy)
        );
        
        TaskPlanner customPlanner = new TaskPlanner(nodes) {
            @Override
            public List<PregelTask> plan(Collection<String> updatedChannels) {
                planCalled[0] = true;
                return expectedTasks;
            }
            
            @Override
            protected List<PregelTask> filter(List<PregelTask> tasks) {
                filterCalled[0] = true;
                return tasks;
            }
            
            @Override
            public List<PregelTask> prioritize(List<PregelTask> tasks) {
                prioritizeCalled[0] = true;
                return tasks;
            }
        };
        
        // Call planAndPrioritize
        List<PregelTask> tasks = customPlanner.planAndPrioritize(updatedChannels);
        
        // Verify that all methods were called
        assertThat(planCalled[0]).isTrue();
        assertThat(filterCalled[0]).isTrue();
        assertThat(prioritizeCalled[0]).isTrue();
        
        // Verify the result
        assertThat(tasks).isEqualTo(expectedTasks);
    }
    
    @Test
    void testFilterRemovesInvalidNodes() {
        TaskPlanner planner = new TaskPlanner(nodes);
        
        List<PregelTask> tasks = new ArrayList<>();
        tasks.add(new PregelTask("node1", null, null));
        tasks.add(new PregelTask("node2", null, null));
        tasks.add(new PregelTask("nonexistent", null, null));
        
        List<PregelTask> filteredTasks = planner.filter(tasks);
        
        assertThat(filteredTasks).hasSize(2);
        assertThat(filteredTasks).extracting(PregelTask::getNode).containsExactlyInAnyOrder("node1", "node2");
    }
    
    @Test
    void testPrioritizeKeepsOrder() {
        TaskPlanner planner = new TaskPlanner(nodes);
        
        List<PregelTask> tasks = new ArrayList<>();
        tasks.add(new PregelTask("node1", null, null));
        tasks.add(new PregelTask("node2", null, null));
        
        List<PregelTask> prioritizedTasks = planner.prioritize(tasks);
        
        // Default implementation should maintain original order
        assertThat(prioritizedTasks).hasSize(2);
        assertThat(prioritizedTasks.get(0).getNode()).isEqualTo("node1");
        assertThat(prioritizedTasks.get(1).getNode()).isEqualTo("node2");
    }
    
    @Test
    void testCustomPrioritization() {
        // Custom planner that reverses the order
        TaskPlanner planner = new TaskPlanner(nodes) {
            @Override
            public List<PregelTask> prioritize(List<PregelTask> tasks) {
                List<PregelTask> prioritized = new ArrayList<>(tasks);
                Collections.reverse(prioritized);
                return prioritized;
            }
        };
        
        List<PregelTask> tasks = new ArrayList<>();
        tasks.add(new PregelTask("node1", null, null));
        tasks.add(new PregelTask("node2", null, null));
        
        List<PregelTask> prioritizedTasks = planner.prioritize(tasks);
        
        // Custom implementation should reverse order
        assertThat(prioritizedTasks).hasSize(2);
        assertThat(prioritizedTasks.get(0).getNode()).isEqualTo("node2");
        assertThat(prioritizedTasks.get(1).getNode()).isEqualTo("node1");
    }
}