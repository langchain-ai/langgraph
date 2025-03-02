package com.langgraph.channels;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class TopicChannelTest {

    @Test
    void testEmptyChannel() {
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class)
            .hasMessageContaining("empty");
    }
    
    @Test
    void testUpdateAndGet() {
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        
        // Update with a single value
        boolean updated = channel.update(Collections.singletonList("test"));
        assertThat(updated).isTrue();
        
        // Verify we can get the value as a list
        assertThat(channel.get()).containsExactly("test");
        
        // Update with multiple values
        updated = channel.update(Arrays.asList("second", "third"));
        assertThat(updated).isTrue();
        
        // Verify all values are accumulated
        assertThat(channel.get()).containsExactly("test", "second", "third");
    }
    
    @Test
    void testEmptyUpdate() {
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        
        // Empty update should return false
        boolean updated = channel.update(Collections.emptyList());
        assertThat(updated).isFalse();
        
        // Channel should still be empty
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testMultipleUpdates() {
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        
        // First update
        channel.update(Collections.singletonList("first"));
        
        // Second update
        channel.update(Collections.singletonList("second"));
        
        // Third update with multiple values
        channel.update(Arrays.asList("third", "fourth"));
        
        // Verify all values accumulated
        assertThat(channel.get()).containsExactly("first", "second", "third", "fourth");
    }
    
    @Test
    void testConsumeWithoutReset() {
        TopicChannel<String> channel = new TopicChannel<>(String.class, false);
        
        // Add some values
        channel.update(Arrays.asList("first", "second"));
        
        // Consume should return false and not change the channel
        boolean consumed = channel.consume();
        assertThat(consumed).isFalse();
        
        // Values should still be present
        assertThat(channel.get()).containsExactly("first", "second");
    }
    
    @Test
    void testConsumeWithReset() {
        TopicChannel<String> channel = new TopicChannel<>(String.class, true);
        
        // Add some values
        channel.update(Arrays.asList("first", "second"));
        
        // Consume should return true and reset the channel
        boolean consumed = channel.consume();
        assertThat(consumed).isTrue();
        
        // Channel should be empty after consuming
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class);
        
        // Add new values after reset
        channel.update(Collections.singletonList("new"));
        
        // Verify only new values are present
        assertThat(channel.get()).containsExactly("new");
    }
    
    @Test
    void testCheckpoint() {
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        channel.update(Arrays.asList("first", "second"));
        
        // Create a checkpoint
        List<String> checkpoint = channel.checkpoint();
        assertThat(checkpoint).containsExactly("first", "second");
        
        // Create a new channel from the checkpoint
        TopicChannel<String> newChannel = (TopicChannel<String>) channel.fromCheckpoint(checkpoint);
        
        // Verify the new channel has the same values
        assertThat(newChannel.get()).containsExactly("first", "second");
        
        // Add more values to the original channel
        channel.update(Collections.singletonList("third"));
        
        // Verify the new channel didn't change
        assertThat(newChannel.get()).containsExactly("first", "second");
        assertThat(channel.get()).containsExactly("first", "second", "third");
    }
    
    @Test
    void testCheckpointWithEmptyList() {
        // Create a channel and update with an empty list (which is a no-op)
        TopicChannel<String> channel = new TopicChannel<>(String.class);
        channel.update(Collections.emptyList());
        
        // Channel should still be empty
        assertThatThrownBy(channel::checkpoint)
            .isInstanceOf(EmptyChannelException.class);
        
        // Now add some values and then create an empty topic
        channel.update(Collections.singletonList("test"));
        List<String> checkpoint = channel.checkpoint();
        
        // Create a new channel with an empty checkpoint
        TopicChannel<String> emptyChannel = (TopicChannel<String>) channel.fromCheckpoint(Collections.emptyList());
        
        // The channel should be initialized but have an empty list
        assertThat(emptyChannel.get()).isEmpty();
    }
    
    @Test
    void testEqualsAndHashCode() {
        TopicChannel<String> channel1 = new TopicChannel<>(String.class);
        TopicChannel<String> channel2 = new TopicChannel<>(String.class);
        
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
        
        // Create channels with different reset behavior
        TopicChannel<String> channel3 = new TopicChannel<>(String.class, true);
        
        // Should not be equal to channel with different reset behavior
        assertThat(channel1).isNotEqualTo(channel3);
    }
}