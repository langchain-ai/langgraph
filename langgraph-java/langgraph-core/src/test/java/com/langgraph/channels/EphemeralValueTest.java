package com.langgraph.channels;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class EphemeralValueTest {

    @Test
    void testEmptyChannel() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class)
            .hasMessageContaining("empty");
    }
    
    @Test
    void testUpdateAndGet() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        
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
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        
        // Empty update should return false
        boolean updated = channel.update(Collections.emptyList());
        assertThat(updated).isFalse();
        
        // Channel should still be empty
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testMultipleValuesThrowsException() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        
        // Multiple values should throw exception
        assertThatThrownBy(() -> channel.update(Arrays.asList("one", "two")))
            .isInstanceOf(InvalidUpdateException.class)
            .hasMessageContaining("only one value");
    }
    
    @Test
    void testCheckpoint() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        channel.update(Collections.singletonList("test"));
        
        // Create a checkpoint - should always be null for ephemeral values
        Void checkpoint = channel.checkpoint();
        assertThat(checkpoint).isNull();
        
        // Create a new channel from the checkpoint
        EphemeralValue<String> newChannel = (EphemeralValue<String>) channel.fromCheckpoint(checkpoint);
        
        // New channel should be empty since ephemeral values don't get checkpointed
        assertThatThrownBy(newChannel::get)
            .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testFromNullCheckpoint() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        channel.update(Collections.singletonList("test"));
        
        // Create a new channel from null checkpoint
        EphemeralValue<String> newChannel = (EphemeralValue<String>) channel.fromCheckpoint(null);
        
        // New channel should be empty
        assertThatThrownBy(newChannel::get)
            .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testEqualsAndHashCode() {
        EphemeralValue<String> channel1 = EphemeralValue.<String>create();
        EphemeralValue<String> channel2 = EphemeralValue.<String>create();
        
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
    
    @Test
    void testNullValue() {
        EphemeralValue<String> channel = EphemeralValue.<String>create();
        
        // Update with null value
        channel.update(Collections.singletonList(null));
        
        // Should return null value
        assertThat(channel.get()).isNull();
    }
}