package com.langgraph.pregel;

import com.langgraph.pregel.channel.ChannelWriteEntry;
import com.langgraph.pregel.retry.RetryPolicy;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class PregelNodeTest {

    /**
     * Simple implementation of PregelExecutable for testing
     */
    private static class TestAction implements PregelExecutable<Object, Object> {
        @Override
        public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
            Map<String, Object> output = new HashMap<>();
            output.put("result", "processed");
            return output;
        }
    }

    @Test
    void testConstructors() {
        // Test with full constructor
        TestAction action = new TestAction();
        List<String> channels = Arrays.asList("channel1", "channel2");
        List<String> triggerChannels = Arrays.asList("trigger1");
        List<ChannelWriteEntry> writers = Arrays.asList(new ChannelWriteEntry("output1"));
        RetryPolicy retryPolicy = RetryPolicy.builder().build();

        PregelNode<Object, Object> node1 = new PregelNode<>(
            "node1", action, channels, triggerChannels, writers, retryPolicy
        );
        
        assertThat(node1.getName()).isEqualTo("node1");
        assertThat(node1.getChannels()).containsExactlyInAnyOrderElementsOf(channels);
        assertThat(node1.getTriggerChannels()).containsExactlyInAnyOrderElementsOf(triggerChannels);
        assertThat(node1.getWriteEntries()).containsExactlyElementsOf(writers);
        assertThat(node1.getRetryPolicy()).isEqualTo(retryPolicy);
    }

    @Test
    void testBuilderPattern() {
        // Test builder with all options
        PregelNode<Object, Object> node = new PregelNode.Builder<>("builder-node", new TestAction())
                .channels("channel1")
                .channels(Arrays.asList("channel2", "channel3"))
                .triggerChannels("triggerChannel")
                .writers("output1")
                .writers(new ChannelWriteEntry("output2"))
                .writers("output3", "output4")
                .build();

        assertThat(node.getName()).isEqualTo("builder-node");
        assertThat(node.getChannels()).containsExactlyInAnyOrder("channel1", "channel2", "channel3");
        assertThat(node.getTriggerChannels()).contains("triggerChannel");
        assertThat(node.getWriters()).containsExactlyInAnyOrder("output1", "output2", "output3", "output4");
    }

    @Test
    void testInputChannels() {
        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                .channels("channel1")
                .channels("channel2")
                .build();

        assertThat(node.readsFrom("channel1")).isTrue();
        assertThat(node.readsFrom("channel2")).isTrue();
        assertThat(node.readsFrom("channel3")).isFalse();
    }

    @Test
    void testTriggerChannels() {
        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                .triggerChannels("triggerChannel")
                .build();

        assertThat(node.isTriggeredBy("triggerChannel")).isTrue();
        assertThat(node.isTriggeredBy("otherTrigger")).isFalse();
    }
    
    @Test
    void testMultipleTriggerChannels() {
        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                .triggerChannels("trigger1")
                .triggerChannels("trigger2")
                .build();

        assertThat(node.isTriggeredBy("trigger1")).isTrue();
        assertThat(node.isTriggeredBy("trigger2")).isTrue();
        assertThat(node.isTriggeredBy("trigger3")).isFalse();
        assertThat(node.getTriggerChannels()).containsExactlyInAnyOrder("trigger1", "trigger2");
    }
    

    @Test
    void testWriters() {
        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                .writers("channel1")
                .writers("channel2")
                .build();

        assertThat(node.canWriteTo("channel1")).isTrue();
        assertThat(node.canWriteTo("channel2")).isTrue();
        assertThat(node.canWriteTo("channel3")).isFalse();
    }

    @Test
    void testWriteEntries() {
        // Test passthrough write entry
        ChannelWriteEntry entry1 = new ChannelWriteEntry("channel1");
        
        // Test explicit value write entry
        ChannelWriteEntry entry2 = new ChannelWriteEntry("channel2", "fixed-value");
        
        // Test write entry with mapping function using the builder
        Function<Object, Object> mapper = value -> "mapped-" + value;
        ChannelWriteEntry entry3 = ChannelWriteEntry.builder("channel3")
                .passthrough()
                .mapper(mapper)
                .skipNone(false)
                .build();

        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                .writers(entry1)
                .writers(entry2)
                .writers(entry3)
                .build();

        // Test retrieving write entries
        Optional<ChannelWriteEntry> foundEntry1 = node.getWriteEntry("channel1");
        assertThat(foundEntry1).isPresent();
        assertThat(foundEntry1.get().getChannel()).isEqualTo("channel1");

        Optional<ChannelWriteEntry> foundEntry2 = node.getWriteEntry("channel2");
        assertThat(foundEntry2).isPresent();
        assertThat(foundEntry2.get().getValue()).isEqualTo("fixed-value");

        Optional<ChannelWriteEntry> foundEntry3 = node.getWriteEntry("channel3");
        assertThat(foundEntry3).isPresent();
        assertThat(foundEntry3.get().hasMapper()).isTrue();
    }

    @Test
    void testProcessOutput() {
        // Setup test node with various write entries
        PregelNode<Object, Object> node = new PregelNode.Builder<>("test", new TestAction())
                // Passthrough entry
                .writers("channel1")
                // Fixed value entry
                .writers(new ChannelWriteEntry("channel2", "fixed-value"))
                // Entry with mapper
                .writers(ChannelWriteEntry.builder("channel3")
                        .passthrough()
                        .mapper(value -> "mapped-" + value)
                        .skipNone(false)
                        .build())
                .build();

        // Create test node output
        Map<String, Object> nodeOutput = new HashMap<>();
        nodeOutput.put("channel1", "value1");
        nodeOutput.put("channel3", "value3");
        nodeOutput.put("ignored", "value-ignored");

        // Process the output
        Map<String, Object> processedOutput = node.processOutput(nodeOutput);

        // Verify processed output
        assertThat(processedOutput).containsEntry("channel1", "value1");
        assertThat(processedOutput).containsEntry("channel2", "fixed-value");
        assertThat(processedOutput).containsEntry("channel3", "mapped-value3");
        assertThat(processedOutput).doesNotContainKey("ignored");
    }

    @Test
    void testProcessOutputWithEmptyWriters() {
        // Node with no explicit write entries should pass all outputs through
        PregelNode<Object, Object> node = new PregelNode<>(
            "test", new TestAction(), Collections.emptyList(), Collections.emptyList(), 
            Collections.emptyList(), null
        );

        Map<String, Object> nodeOutput = new HashMap<>();
        nodeOutput.put("channel1", "value1");
        nodeOutput.put("channel2", "value2");

        Map<String, Object> processedOutput = node.processOutput(nodeOutput);

        // All outputs should be passed through
        assertThat(processedOutput).isEqualTo(nodeOutput);
    }

    @Test
    void testNodeEquality() {
        PregelNode<Object, Object> node1 = new PregelNode<>(
            "same-name", new TestAction(), Collections.emptyList(), Collections.emptyList(),
            Collections.emptyList(), null
        );
        PregelNode<Object, Object> node2 = new PregelNode<>(
            "same-name", new TestAction(), Collections.emptyList(), Collections.emptyList(),
            Collections.emptyList(), null
        );
        PregelNode<Object, Object> node3 = new PregelNode<>(
            "different-name", new TestAction(), Collections.emptyList(), Collections.emptyList(),
            Collections.emptyList(), null
        );

        // Nodes with same name should be equal
        assertThat(node1).isEqualTo(node2);
        assertThat(node1.hashCode()).isEqualTo(node2.hashCode());

        // Nodes with different names should not be equal
        assertThat(node1).isNotEqualTo(node3);
    }

    @Test
    void testInvalidConstruction() {
        // Test null name
        assertThatThrownBy(() -> new PregelNode<>(
                null, new TestAction(), Collections.emptyList(), Collections.emptyList(),
                Collections.emptyList(), null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("name cannot be null");

        // Test empty name
        assertThatThrownBy(() -> new PregelNode<>(
                "", new TestAction(), Collections.emptyList(), Collections.emptyList(),
                Collections.emptyList(), null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("name cannot be null or empty");

        // Test null action
        assertThatThrownBy(() -> new PregelNode<>(
                "test", null, Collections.emptyList(), Collections.emptyList(),
                Collections.emptyList(), null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Action cannot be null");
    }
}