package com.langgraph.channels;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class LastValueTest {

    @Test
    void testEmptyChannel() {
        LastValue<String> channel = new LastValue<>(String.class);
        // With Python compatibility, uninitialized channels return null rather than throwing
        assertThat(channel.get()).isNull();
    }
    
    @Test
    void testUpdateAndGet() {
        LastValue<String> channel = new LastValue<>(String.class);
        
        // Update with a value
        boolean updated = channel.update(Collections.singletonList("test"));
        assertThat(updated).isTrue();
        
        // Verify we can get the value
        assertThat(channel.get()).isEqualTo("test");
        
        // Update with another value
        updated = channel.update(Collections.singletonList("updated"));
        assertThat(updated).isTrue();
        
        // Verify the value was updated
        assertThat(channel.get()).isEqualTo("updated");
    }
    
    @Test
    void testEmptyUpdate() {
        LastValue<String> channel = new LastValue<>(String.class);
        
        // Empty update should return false
        boolean updated = channel.update(Collections.emptyList());
        assertThat(updated).isFalse();
        
        // Channel should still be uninitialized (returns null with Python compatibility)
        assertThat(channel.get()).isNull();
    }
    
    @Test
    void testMultipleValuesThrowsException() {
        LastValue<String> channel = new LastValue<>(String.class);
        
        // Multiple values should throw exception
        assertThatThrownBy(() -> channel.update(Arrays.asList("one", "two")))
            .isInstanceOf(InvalidUpdateException.class)
            .hasMessageContaining("only one value");
    }
    
    @Test
    void testCheckpoint() {
        LastValue<String> channel = new LastValue<>(String.class);
        channel.update(Collections.singletonList("test"));
        
        // Create a checkpoint
        String checkpoint = channel.checkpoint();
        assertThat(checkpoint).isEqualTo("test");
        
        // Create a new channel from the checkpoint
        LastValue<String> newChannel = (LastValue<String>) channel.fromCheckpoint(checkpoint);
        
        // Verify the new channel has the same value
        assertThat(newChannel.get()).isEqualTo("test");
    }
    
    @Test
    void testCheckpointWithNullValue() {
        LastValue<String> channel = new LastValue<>(String.class);
        channel.update(Collections.singletonList(null));
        
        // Create a checkpoint
        String checkpoint = channel.checkpoint();
        assertThat(checkpoint).isNull();
        
        // Create a new channel from the checkpoint
        LastValue<String> newChannel = (LastValue<String>) channel.fromCheckpoint(checkpoint);
        
        // Verify the new channel has the same null value
        assertThat(newChannel.get()).isNull();
    }
    
    @Test
    void testEqualsAndHashCode() {
        LastValue<String> channel1 = new LastValue<>(String.class);
        LastValue<String> channel2 = new LastValue<>(String.class);
        
        // Initially equal
        assertThat(channel1).isEqualTo(channel2);
        assertThat(channel1.hashCode()).isEqualTo(channel2.hashCode());
        
        // Update one channel
        channel1.update(Collections.singletonList("test"));
        
        // No longer equal
        assertThat(channel1).isNotEqualTo(channel2);
        
        // Update second channel with same value
        channel2.update(Collections.singletonList("test"));
        
        // Equal again
        assertThat(channel1).isEqualTo(channel2);
        assertThat(channel1.hashCode()).isEqualTo(channel2.hashCode());
    }
}