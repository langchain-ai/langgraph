package com.langgraph.channels;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.function.BinaryOperator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class BinaryOperatorChannelTest {

    @Test
    void testEmptyChannel() {
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(Integer::sum, 0);
        
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class)
            .hasMessageContaining("empty");
    }
    
    @Test
    void testSumOperator() {
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(Integer::sum, 0);
        
        // Initial update
        boolean updated = channel.update(Collections.singletonList(5));
        assertThat(updated).isTrue();
        assertThat(channel.get()).isEqualTo(5);
        
        // Add more values
        updated = channel.update(Arrays.asList(10, 7, 3));
        assertThat(updated).isTrue();
        assertThat(channel.get()).isEqualTo(25); // 5 + 10 + 7 + 3 = 25
    }
    
    @Test
    void testMaxOperator() {
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(Integer::max, Integer.MIN_VALUE);
        
        // Initial update
        channel.update(Collections.singletonList(5));
        assertThat(channel.get()).isEqualTo(5);
        
        // Add higher values
        channel.update(Arrays.asList(10, 7));
        assertThat(channel.get()).isEqualTo(10);
        
        // Add lower values
        channel.update(Collections.singletonList(3));
        assertThat(channel.get()).isEqualTo(10); // Max is still 10
    }
    
    @Test
    void testStringConcatenation() {
        BinaryOperator<String> concat = (a, b) -> a + b;
        BinaryOperatorChannel<String> channel = BinaryOperatorChannel.create(concat, "");
        
        // Initial update
        channel.update(Collections.singletonList("Hello"));
        assertThat(channel.get()).isEqualTo("Hello");
        
        // Add more values
        channel.update(Arrays.asList(", ", "World", "!"));
        assertThat(channel.get()).isEqualTo("Hello, World!");
    }
    
    @Test
    void testEmptyUpdate() {
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(Integer::sum, 0);
        
        // Empty update should return false
        boolean updated = channel.update(Collections.emptyList());
        assertThat(updated).isFalse();
        
        // Channel should still be empty
        assertThatThrownBy(channel::get)
            .isInstanceOf(EmptyChannelException.class);
    }
    
    @Test
    void testUpdateOrder() {
        // Using subtraction to check order (not commutative)
        BinaryOperator<Integer> subtract = (a, b) -> a - b;
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(subtract, 100);
        
        // Subtract values from 100
        channel.update(Arrays.asList(20, 30));
        
        // Result should be 100 - 20 - 30 = 50
        assertThat(channel.get()).isEqualTo(50);
    }
    
    @Test
    void testCheckpoint() {
        BinaryOperatorChannel<Integer> channel = BinaryOperatorChannel.create(Integer::sum, 0);
        
        // Update the channel
        channel.update(Arrays.asList(5, 10, 15));
        
        // Create a checkpoint
        Integer checkpoint = channel.checkpoint();
        assertThat(checkpoint).isEqualTo(30);
        
        // Create a new channel from the checkpoint
        BinaryOperatorChannel<Integer> newChannel = 
            (BinaryOperatorChannel<Integer>) channel.fromCheckpoint(checkpoint);
        
        // Verify the new channel has the same accumulated value
        assertThat(newChannel.get()).isEqualTo(30);
        
        // Add more to the original
        channel.update(Collections.singletonList(20));
        assertThat(channel.get()).isEqualTo(50);
        
        // New channel should be unchanged
        assertThat(newChannel.get()).isEqualTo(30);
        
        // Add to the new channel
        newChannel.update(Collections.singletonList(5));
        assertThat(newChannel.get()).isEqualTo(35);
    }
    
    @Test
    void testUtilityMethods() {
        // Test the utility method for Integer adder
        BinaryOperatorChannel<Integer> intAdder = Channels.integerAdder("counter");
        intAdder.update(Arrays.asList(1, 2, 3));
        assertThat(intAdder.get()).isEqualTo(6);
        
        // Test the utility method for Long adder
        BinaryOperatorChannel<Long> longAdder = Channels.longAdder("longCounter");
        longAdder.update(Arrays.asList(1L, 2L, 3L));
        assertThat(longAdder.get()).isEqualTo(6L);
        
        // Test the utility method for Double adder
        BinaryOperatorChannel<Double> doubleAdder = Channels.doubleAdder("doubleCounter");
        doubleAdder.update(Arrays.asList(1.5, 2.5));
        assertThat(doubleAdder.get()).isEqualTo(4.0);
        
        // Test the utility method for Integer max
        BinaryOperatorChannel<Integer> intMax = Channels.integerMax("maxValue");
        intMax.update(Arrays.asList(5, 10, 3));
        assertThat(intMax.get()).isEqualTo(10);
    }
}