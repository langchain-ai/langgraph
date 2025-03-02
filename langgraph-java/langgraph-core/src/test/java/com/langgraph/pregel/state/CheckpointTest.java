package com.langgraph.pregel.state;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class CheckpointTest {
    
    @Test
    void testEmptyCheckpoint() {
        Checkpoint checkpoint = new Checkpoint();
        
        assertThat(checkpoint.isEmpty()).isTrue();
        assertThat(checkpoint.size()).isEqualTo(0);
        assertThat(checkpoint.getValues()).isEmpty();
        assertThat(checkpoint.containsChannel("channel1")).isFalse();
        assertThat(checkpoint.getValue("channel1")).isNull();
    }
    
    @Test
    void testCheckpointWithValues() {
        Map<String, Object> values = new HashMap<>();
        values.put("channel1", "value1");
        values.put("channel2", 42);
        
        Checkpoint checkpoint = new Checkpoint(values);
        
        assertThat(checkpoint.isEmpty()).isFalse();
        assertThat(checkpoint.size()).isEqualTo(2);
        assertThat(checkpoint.getValues()).hasSize(2);
        assertThat(checkpoint.containsChannel("channel1")).isTrue();
        assertThat(checkpoint.containsChannel("channel2")).isTrue();
        assertThat(checkpoint.containsChannel("channel3")).isFalse();
        assertThat(checkpoint.getValue("channel1")).isEqualTo("value1");
        assertThat(checkpoint.getValue("channel2")).isEqualTo(42);
        assertThat(checkpoint.getValue("channel3")).isNull();
    }
    
    @Test
    void testConstructorWithNullValues() {
        Checkpoint checkpoint = new Checkpoint(null);
        
        assertThat(checkpoint.isEmpty()).isTrue();
        assertThat(checkpoint.size()).isEqualTo(0);
    }
    
    @Test
    void testUpdate() {
        Checkpoint checkpoint = new Checkpoint();
        
        // Initial state
        assertThat(checkpoint.isEmpty()).isTrue();
        
        // Update with new values
        Map<String, Object> values = new HashMap<>();
        values.put("channel1", "value1");
        values.put("channel2", 42);
        
        checkpoint.update(values);
        
        assertThat(checkpoint.isEmpty()).isFalse();
        assertThat(checkpoint.size()).isEqualTo(2);
        assertThat(checkpoint.getValue("channel1")).isEqualTo("value1");
        assertThat(checkpoint.getValue("channel2")).isEqualTo(42);
        
        // Update should overwrite all previous values
        Map<String, Object> newValues = new HashMap<>();
        newValues.put("channel3", "value3");
        
        checkpoint.update(newValues);
        
        assertThat(checkpoint.size()).isEqualTo(1);
        assertThat(checkpoint.containsChannel("channel1")).isFalse();
        assertThat(checkpoint.containsChannel("channel2")).isFalse();
        assertThat(checkpoint.containsChannel("channel3")).isTrue();
        assertThat(checkpoint.getValue("channel3")).isEqualTo("value3");
    }
    
    @Test
    void testUpdateWithNull() {
        Checkpoint checkpoint = new Checkpoint();
        
        assertThatThrownBy(() -> checkpoint.update(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null");
    }
    
    @Test
    void testUpdateChannel() {
        Checkpoint checkpoint = new Checkpoint();
        
        // Update single channel
        checkpoint.updateChannel("channel1", "value1");
        
        assertThat(checkpoint.isEmpty()).isFalse();
        assertThat(checkpoint.size()).isEqualTo(1);
        assertThat(checkpoint.getValue("channel1")).isEqualTo("value1");
        
        // Update existing channel
        checkpoint.updateChannel("channel1", "newValue");
        
        assertThat(checkpoint.size()).isEqualTo(1);
        assertThat(checkpoint.getValue("channel1")).isEqualTo("newValue");
        
        // Add another channel
        checkpoint.updateChannel("channel2", 42);
        
        assertThat(checkpoint.size()).isEqualTo(2);
        assertThat(checkpoint.getValue("channel1")).isEqualTo("newValue");
        assertThat(checkpoint.getValue("channel2")).isEqualTo(42);
        
        // Remove a channel by setting its value to null
        checkpoint.updateChannel("channel1", null);
        
        assertThat(checkpoint.size()).isEqualTo(1);
        assertThat(checkpoint.containsChannel("channel1")).isFalse();
        assertThat(checkpoint.containsChannel("channel2")).isTrue();
    }
    
    @Test
    void testUpdateChannelWithInvalidName() {
        Checkpoint checkpoint = new Checkpoint();
        
        assertThatThrownBy(() -> checkpoint.updateChannel(null, "value"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        
        assertThatThrownBy(() -> checkpoint.updateChannel("", "value"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
    }
    
    @Test
    void testWithUpdates() {
        Map<String, Object> initialValues = new HashMap<>();
        initialValues.put("channel1", "value1");
        initialValues.put("channel2", 42);
        
        Checkpoint checkpoint = new Checkpoint(initialValues);
        
        // Update with null or empty map should return the same checkpoint
        Checkpoint sameCheckpoint = checkpoint.withUpdates(null);
        assertThat(sameCheckpoint).isSameAs(checkpoint);
        
        sameCheckpoint = checkpoint.withUpdates(Collections.emptyMap());
        assertThat(sameCheckpoint).isSameAs(checkpoint);
        
        // Create new checkpoint with updates
        Map<String, Object> updates = new HashMap<>();
        updates.put("channel2", 99);  // Update existing channel
        updates.put("channel3", "value3");  // Add new channel
        
        Checkpoint updatedCheckpoint = checkpoint.withUpdates(updates);
        
        // Original should be unchanged
        assertThat(checkpoint.size()).isEqualTo(2);
        assertThat(checkpoint.getValue("channel1")).isEqualTo("value1");
        assertThat(checkpoint.getValue("channel2")).isEqualTo(42);
        assertThat(checkpoint.containsChannel("channel3")).isFalse();
        
        // New checkpoint should have updates
        assertThat(updatedCheckpoint).isNotSameAs(checkpoint);
        assertThat(updatedCheckpoint.size()).isEqualTo(3);
        assertThat(updatedCheckpoint.getValue("channel1")).isEqualTo("value1");
        assertThat(updatedCheckpoint.getValue("channel2")).isEqualTo(99);
        assertThat(updatedCheckpoint.getValue("channel3")).isEqualTo("value3");
    }
    
    @Test
    void testSubset() {
        Map<String, Object> values = new HashMap<>();
        values.put("channel1", "value1");
        values.put("channel2", 42);
        values.put("channel3", "value3");
        
        Checkpoint checkpoint = new Checkpoint(values);
        
        // Subset with null should return empty checkpoint
        Checkpoint nullSubset = checkpoint.subset(null);
        assertThat(nullSubset.isEmpty()).isTrue();
        
        // Subset with empty list should return empty checkpoint
        Checkpoint emptySubset = checkpoint.subset(Collections.emptyList());
        assertThat(emptySubset.isEmpty()).isTrue();
        
        // Subset with selected channels
        List<String> channels = Arrays.asList("channel1", "channel3", "nonexistent");
        Checkpoint subsetCheckpoint = checkpoint.subset(channels);
        
        assertThat(subsetCheckpoint.size()).isEqualTo(2);
        assertThat(subsetCheckpoint.containsChannel("channel1")).isTrue();
        assertThat(subsetCheckpoint.containsChannel("channel2")).isFalse();
        assertThat(subsetCheckpoint.containsChannel("channel3")).isTrue();
        assertThat(subsetCheckpoint.getValue("channel1")).isEqualTo("value1");
        assertThat(subsetCheckpoint.getValue("channel3")).isEqualTo("value3");
    }
    
    @Test
    void testEqualsAndHashCode() {
        Map<String, Object> values1 = new HashMap<>();
        values1.put("channel1", "value1");
        values1.put("channel2", 42);
        
        Map<String, Object> values2 = new HashMap<>();
        values2.put("channel1", "value1");
        values2.put("channel2", 42);
        
        Map<String, Object> values3 = new HashMap<>();
        values3.put("channel1", "value1");
        values3.put("channel2", 99);
        
        Checkpoint checkpoint1 = new Checkpoint(values1);
        Checkpoint checkpoint2 = new Checkpoint(values2);
        Checkpoint checkpoint3 = new Checkpoint(values3);
        Checkpoint checkpoint4 = new Checkpoint();
        
        // Same values should be equal
        assertThat(checkpoint1).isEqualTo(checkpoint2);
        assertThat(checkpoint1.hashCode()).isEqualTo(checkpoint2.hashCode());
        
        // Different values should not be equal
        assertThat(checkpoint1).isNotEqualTo(checkpoint3);
        
        // Empty checkpoint should not equal non-empty
        assertThat(checkpoint1).isNotEqualTo(checkpoint4);
        
        // Should not equal null or other objects
        assertThat(checkpoint1).isNotEqualTo(null);
        assertThat(checkpoint1).isNotEqualTo("not a checkpoint");
    }
    
    @Test
    void testToString() {
        Checkpoint emptyCheckpoint = new Checkpoint();
        assertThat(emptyCheckpoint.toString()).contains("channelCount=0");
        
        Map<String, Object> values = new HashMap<>();
        values.put("channel1", "value1");
        values.put("channel2", 42);
        
        Checkpoint checkpoint = new Checkpoint(values);
        assertThat(checkpoint.toString()).contains("channelCount=2");
    }
}