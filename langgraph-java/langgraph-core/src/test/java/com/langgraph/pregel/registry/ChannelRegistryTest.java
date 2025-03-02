package com.langgraph.pregel.registry;

import com.langgraph.channels.BaseChannel;
import com.langgraph.channels.EmptyChannelException;
import com.langgraph.channels.LastValue;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class ChannelRegistryTest {
    
    // Test-specific channel that overrides getValue() and checkpoint() to avoid EmptyChannelException
    private static class TestChannel<T> extends LastValue<T> {
        private boolean initialized = false;
        private T value;
        
        public TestChannel(Class<T> valueType, String key) {
            super(valueType, key);
        }
        
        @Override
        public Object getValue() {
            try {
                return get();
            } catch (EmptyChannelException e) {
                // Return null instead of throwing
                return null;
            }
        }
        
        @Override
        public T checkpoint() {
            try {
                return get();
            } catch (EmptyChannelException e) {
                // Return null instead of throwing
                return null;
            }
        }
        
        @Override
        public boolean update(List<T> values) {
            if (values.isEmpty()) {
                return false;
            }
            
            value = values.get(0);
            initialized = true;
            return true;
        }
        
        @Override
        public T get() throws EmptyChannelException {
            if (!initialized) {
                throw new EmptyChannelException("TestChannel at key '" + key + "' is empty (never updated)");
            }
            return value;
        }
    }
    
    private BaseChannel<String, String, String> channel1;
    private BaseChannel<Integer, Integer, Integer> channel2;
    private BaseChannel<Double, Double, Double> channel3;
    
    @BeforeEach
    void setUp() {
        // Create TestChannel instances for testing
        channel1 = new TestChannel<>(String.class, "channel1");
        channel2 = new TestChannel<>(Integer.class, "channel2");
        channel3 = new TestChannel<>(Double.class, "channel3");
    }
    
    @Test
    void testEmptyRegistry() {
        ChannelRegistry registry = new ChannelRegistry();
        
        assertThat(registry.size()).isEqualTo(0);
        assertThat(registry.getAll()).isEmpty();
        assertThat(registry.getNames()).isEmpty();
        assertThatThrownBy(() -> registry.get("nonexistent"))
                .isInstanceOf(NoSuchElementException.class)
                .hasMessageContaining("nonexistent");
    }
    
    @Test
    void testRegisterChannel() {
        ChannelRegistry registry = new ChannelRegistry();
        
        registry.register("channel1", channel1);
        
        assertThat(registry.size()).isEqualTo(1);
        assertThat(registry.contains("channel1")).isTrue();
        assertThat(registry.get("channel1")).isSameAs(channel1);
        assertThat(registry.getNames()).containsExactly("channel1");
    }
    
    @Test
    void testRegisterDuplicateChannel() {
        ChannelRegistry registry = new ChannelRegistry();
        
        registry.register("channel1", channel1);
        
        assertThatThrownBy(() -> registry.register("channel1", channel2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("already registered");
    }
    
    @Test
    void testRegisterInvalidChannel() {
        ChannelRegistry registry = new ChannelRegistry();
        
        assertThatThrownBy(() -> registry.register("", channel1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        
        assertThatThrownBy(() -> registry.register(null, channel1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null or empty");
        
        assertThatThrownBy(() -> registry.register("channel1", null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cannot be null");
    }
    
    @Test
    void testRegisterAllChannels() {
        ChannelRegistry registry = new ChannelRegistry();
        
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("channel1", channel1);
        channels.put("channel2", channel2);
        
        registry.registerAll(channels);
        
        assertThat(registry.size()).isEqualTo(2);
        assertThat(registry.contains("channel1")).isTrue();
        assertThat(registry.contains("channel2")).isTrue();
        assertThat(registry.get("channel1")).isSameAs(channel1);
        assertThat(registry.get("channel2")).isSameAs(channel2);
    }
    
    @Test
    void testConstructorWithMap() {
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("channel1", channel1);
        channels.put("channel2", channel2);
        
        ChannelRegistry registry = new ChannelRegistry(channels);
        
        assertThat(registry.size()).isEqualTo(2);
        assertThat(registry.contains("channel1")).isTrue();
        assertThat(registry.contains("channel2")).isTrue();
        assertThat(registry.getNames()).containsExactlyInAnyOrder("channel1", "channel2");
    }
    
    @Test
    void testRemoveChannel() {
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("channel1", channel1);
        channels.put("channel2", channel2);
        
        ChannelRegistry registry = new ChannelRegistry(channels);
        
        registry.remove("channel1");
        
        assertThat(registry.size()).isEqualTo(1);
        assertThat(registry.contains("channel1")).isFalse();
        assertThat(registry.contains("channel2")).isTrue();
        assertThat(registry.getNames()).containsExactly("channel2");
    }
    
    @Test
    void testUpdateChannel() {
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        
        String value = "test_value";
        
        boolean updated = registry.update("channel1", value);
        
        assertThat(updated).isTrue();
        // Verify the actual state instead of mock interaction
        assertThat(channel1.get()).isEqualTo(value);
    }
    
    @Test
    void testUpdateNonExistentChannel() {
        ChannelRegistry registry = new ChannelRegistry();
        
        assertThatThrownBy(() -> registry.update("nonexistent", "value"))
                .isInstanceOf(NoSuchElementException.class)
                .hasMessageContaining("nonexistent");
    }
    
    @Test
    void testUpdateAll() {
        ChannelRegistry registry = new ChannelRegistry();
        
        // For this test, create a special channel2 that returns false on update
        BaseChannel<String, String, String> nonUpdatingChannel = new LastValue<>(String.class, "channel2") {
            @Override
            public boolean update(List<String> values) {
                // Override update to always return false
                return false;
            }
        };
        
        registry.register("channel1", channel1);
        registry.register("channel2", nonUpdatingChannel);
        registry.register("channel3", channel3);
        
        Map<String, Object> updates = new HashMap<>();
        updates.put("channel1", "value1");
        updates.put("channel2", "value2");
        updates.put("channel3", 3.14);
        updates.put("nonexistent", "value4"); // This should be ignored
        
        Set<String> updatedChannels = registry.updateAll(updates);
        
        assertThat(updatedChannels).containsExactlyInAnyOrder("channel1", "channel3");
        
        // Verify the actual state instead of mock interactions
        assertThat(channel1.get()).isEqualTo("value1");
        assertThatThrownBy(() -> nonUpdatingChannel.get())
                .isInstanceOf(EmptyChannelException.class);
        assertThat(channel3.get()).isEqualTo(3.14);
    }
    
    @Test
    void testUpdateAllWithEmptyMap() {
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        
        Set<String> updatedChannels = registry.updateAll(null);
        assertThat(updatedChannels).isEmpty();
        
        updatedChannels = registry.updateAll(Collections.emptyMap());
        assertThat(updatedChannels).isEmpty();
        
        // Verify the channel hasn't been updated
        assertThatThrownBy(() -> channel1.get())
                .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testCollectValues() {
        // Use our spied channels - they will return null instead of throwing EmptyChannelException
        channel1.update(Collections.singletonList("value1"));
        // channel2 is left uninitialized
        channel3.update(Collections.singletonList(42.0));
        
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        registry.register("channel2", channel2);
        registry.register("channel3", channel3);
        
        Map<String, Object> values = registry.collectValues();
        
        assertThat(values).hasSize(2);
        assertThat(values).containsEntry("channel1", "value1");
        assertThat(values).containsEntry("channel3", 42.0);
        assertThat(values).doesNotContainKey("channel2");
    }
    
    @Test
    void testCheckpoint() {
        // Use our spied channels - they will return null instead of throwing EmptyChannelException
        channel1.update(Collections.singletonList("checkpoint1"));
        // channel2 is left uninitialized
        channel3.update(Collections.singletonList(42.0));
        
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        registry.register("channel2", channel2);
        registry.register("channel3", channel3);
        
        Map<String, Object> checkpointData = registry.checkpoint();
        
        assertThat(checkpointData).hasSize(2);
        assertThat(checkpointData).containsEntry("channel1", "checkpoint1");
        assertThat(checkpointData).containsEntry("channel3", 42.0);
        assertThat(checkpointData).doesNotContainKey("channel2");
    }
    
    @Test
    void testRestoreFromCheckpoint() {
        // Create new TestChannel instances
        TestChannel<String> stringChannel = new TestChannel<>(String.class, "stringChannel");
        TestChannel<Integer> intChannel = new TestChannel<>(Integer.class, "intChannel");
        
        // Override fromCheckpoint to make it work for testing
        TestChannel<String> testChannel1 = new TestChannel<String>(String.class, "channel1") {
            @Override
            public BaseChannel<String, String, String> fromCheckpoint(String checkpoint) {
                // Just update the current instance instead of creating a new one
                update(Collections.singletonList(checkpoint));
                return this;
            }
        };
        
        TestChannel<Integer> testChannel2 = new TestChannel<Integer>(Integer.class, "channel2") {
            @Override
            public BaseChannel<Integer, Integer, Integer> fromCheckpoint(Integer checkpoint) {
                // Just update the current instance instead of creating a new one
                update(Collections.singletonList(checkpoint));
                return this;
            }
        };
        
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", testChannel1);
        registry.register("channel2", testChannel2);
        
        Map<String, Object> checkpointData = new HashMap<>();
        checkpointData.put("channel1", "checkpoint1");
        checkpointData.put("channel2", 42);
        checkpointData.put("nonexistent", "checkpoint3"); // This should be ignored
        
        registry.restoreFromCheckpoint(checkpointData);
        
        // Verify that the channels have been restored with the values
        assertThat(testChannel1.get()).isEqualTo("checkpoint1");
        assertThat(testChannel2.get()).isEqualTo(42);
    }
    
    @Test
    void testRestoreFromCheckpointWithEmptyMap() {
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        
        // This should not throw an exception
        registry.restoreFromCheckpoint(null);
        registry.restoreFromCheckpoint(Collections.emptyMap());
        
        // Verify the channel is still empty
        assertThatThrownBy(() -> ((TestChannel<String>)channel1).get())
                .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testResetUpdated() {
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        registry.register("channel2", channel2);
        
        // Update channels
        channel1.update(Collections.singletonList("value1"));
        channel2.update(Collections.singletonList(42));
        
        // Now reset
        registry.resetUpdated();
        
        // Verify the channels still have their values after reset
        assertThat(((TestChannel<String>)channel1).get()).isEqualTo("value1");
        assertThat(((TestChannel<Integer>)channel2).get()).isEqualTo(42);
    }
    
    @Test
    void testSubset() {
        Map<String, BaseChannel> channels = new HashMap<>();
        channels.put("channel1", channel1);
        channels.put("channel2", channel2);
        channels.put("channel3", channel3);
        
        ChannelRegistry registry = new ChannelRegistry(channels);
        
        ChannelRegistry subset = registry.subset(Arrays.asList("channel1", "channel3", "nonexistent"));
        
        assertThat(subset.size()).isEqualTo(2);
        assertThat(subset.contains("channel1")).isTrue();
        assertThat(subset.contains("channel3")).isTrue();
        assertThat(subset.contains("channel2")).isFalse();
        assertThat(subset.contains("nonexistent")).isFalse();
        
        assertThat(subset.get("channel1")).isSameAs(channel1);
        assertThat(subset.get("channel3")).isSameAs(channel3);
    }
    
    @Test
    void testSubsetWithNull() {
        ChannelRegistry registry = new ChannelRegistry();
        registry.register("channel1", channel1);
        
        ChannelRegistry subset = registry.subset(null);
        
        assertThat(subset.size()).isEqualTo(0);
    }
}