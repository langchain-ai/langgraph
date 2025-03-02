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
    private static class TestAction implements PregelExecutable {
        @Override
        public Map<String, Object> execute(Map<String, Object> inputs, Map<String, Object> context) {
            Map<String, Object> output = new HashMap<>();
            output.put("result", "processed");
            return output;
        }
    }

    @Test
    void testConstructors() {
        // Test minimal constructor
        PregelNode node1 = new PregelNode("node1", new TestAction());
        assertThat(node1.getName()).isEqualTo("node1");
        assertThat(node1.getSubscribe()).isEmpty();
        assertThat(node1.getTrigger()).isNull();
        assertThat(node1.getWriteEntries()).isEmpty();
        assertThat(node1.getRetryPolicy()).isNull();

        // Test constructor with subscriptions
        List<String> subscriptions = Arrays.asList("channel1", "channel2");
        PregelNode node2 = new PregelNode("node2", new TestAction(), subscriptions);
        assertThat(node2.getName()).isEqualTo("node2");
        assertThat(node2.getSubscribe()).containsExactlyInAnyOrderElementsOf(subscriptions);
        assertThat(node2.getTrigger()).isNull();
        assertThat(node2.getWriteEntries()).isEmpty();
    }

    @Test
    void testBuilderPattern() {
        // Test builder with all options
        PregelNode node = new PregelNode.Builder("builder-node", new TestAction())
                .subscribe("channel1")
                .subscribeAll(Arrays.asList("channel2", "channel3"))
                .trigger("triggerChannel")
                .writer("output1")
                .writer(new ChannelWriteEntry("output2"))
                .writeAllNames(Arrays.asList("output3", "output4"))
                .build();

        assertThat(node.getName()).isEqualTo("builder-node");
        assertThat(node.getSubscribe()).containsExactlyInAnyOrder("channel1", "channel2", "channel3");
        assertThat(node.getTrigger()).isEqualTo("triggerChannel");
        assertThat(node.getWriters()).containsExactlyInAnyOrder("output1", "output2", "output3", "output4");
    }

    @Test
    void testSubscriptions() {
        PregelNode node = new PregelNode.Builder("test", new TestAction())
                .subscribe("channel1")
                .subscribe("channel2")
                .build();

        assertThat(node.subscribesTo("channel1")).isTrue();
        assertThat(node.subscribesTo("channel2")).isTrue();
        assertThat(node.subscribesTo("channel3")).isFalse();
    }

    @Test
    void testTriggers() {
        PregelNode node = new PregelNode.Builder("test", new TestAction())
                .trigger("triggerChannel")
                .build();

        assertThat(node.hasTrigger("triggerChannel")).isTrue();
        assertThat(node.hasTrigger("otherTrigger")).isFalse();
    }

    @Test
    void testWriters() {
        PregelNode node = new PregelNode.Builder("test", new TestAction())
                .writer("channel1")
                .writer("channel2")
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

        PregelNode node = new PregelNode.Builder("test", new TestAction())
                .writer(entry1)
                .writer(entry2)
                .writer(entry3)
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
        PregelNode node = new PregelNode.Builder("test", new TestAction())
                // Passthrough entry
                .writer("channel1")
                // Fixed value entry
                .writer(new ChannelWriteEntry("channel2", "fixed-value"))
                // Entry with mapper
                .writer(ChannelWriteEntry.builder("channel3")
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
        PregelNode node = new PregelNode("test", new TestAction());

        Map<String, Object> nodeOutput = new HashMap<>();
        nodeOutput.put("channel1", "value1");
        nodeOutput.put("channel2", "value2");

        Map<String, Object> processedOutput = node.processOutput(nodeOutput);

        // All outputs should be passed through
        assertThat(processedOutput).isEqualTo(nodeOutput);
    }

    @Test
    void testNodeEquality() {
        PregelNode node1 = new PregelNode("same-name", new TestAction());
        PregelNode node2 = new PregelNode("same-name", new TestAction());
        PregelNode node3 = new PregelNode("different-name", new TestAction());

        // Nodes with same name should be equal
        assertThat(node1).isEqualTo(node2);
        assertThat(node1.hashCode()).isEqualTo(node2.hashCode());

        // Nodes with different names should not be equal
        assertThat(node1).isNotEqualTo(node3);
    }

    @Test
    void testInvalidConstruction() {
        // Test null name
        assertThatThrownBy(() -> new PregelNode(null, new TestAction()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("name cannot be null");

        // Test empty name
        assertThatThrownBy(() -> new PregelNode("", new TestAction()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("name cannot be null or empty");

        // Test null action
        assertThatThrownBy(() -> new PregelNode("test", null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Action cannot be null");
    }
}