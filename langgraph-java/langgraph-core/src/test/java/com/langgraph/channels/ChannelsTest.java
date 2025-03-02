package com.langgraph.channels;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BinaryOperator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class ChannelsTest {

    @Test
    void testLastValueFactory() {
        // Create channel using factory
        LastValue<String> channel = Channels.lastValue(String.class);
        
        // Update the channel
        channel.update(Collections.singletonList("test"));
        
        // Verify it works
        assertThat(channel.get()).isEqualTo("test");
        
        // Create channel with key
        LastValue<String> namedChannel = Channels.lastValue(String.class, "input");
        assertThat(namedChannel.getKey()).isEqualTo("input");
    }
    
    @Test
    void testTopicFactory() {
        // Create topic channel using factory
        TopicChannel<String> channel = Channels.topic(String.class);
        
        // Update with values
        channel.update(Arrays.asList("one", "two"));
        
        // Verify it works
        assertThat(channel.get()).containsExactly("one", "two");
        
        // Create reset-on-consume topic
        TopicChannel<String> resetChannel = Channels.topic(String.class, true);
        resetChannel.update(Collections.singletonList("test"));
        
        // Consume should reset the channel
        boolean consumed = resetChannel.consume();
        assertThat(consumed).isTrue();
        
        // Channel should be empty but not throw with Python compatibility
        assertThat(resetChannel.get()).isEmpty();
        
        // Create with key
        TopicChannel<String> namedChannel = Channels.topic(String.class, "messages", false);
        assertThat(namedChannel.getKey()).isEqualTo("messages");
    }
    
    @Test
    void testBinaryOperatorFactory() {
        // Create a binary operator channel using factory
        BinaryOperator<Integer> sum = Integer::sum;
        BinaryOperatorChannel<Integer> channel = Channels.binaryOperator(Integer.class, sum, 0);
        
        // Update with values
        channel.update(Arrays.asList(1, 2, 3));
        
        // Verify it works
        assertThat(channel.get()).isEqualTo(6);
        
        // Create with key
        BinaryOperatorChannel<Integer> namedChannel = 
            Channels.binaryOperator(Integer.class, "counter", sum, 0);
        assertThat(namedChannel.getKey()).isEqualTo("counter");
    }
    
    @Test
    void testEphemeralFactory() {
        // Create ephemeral channel using factory
        EphemeralValue<String> channel = Channels.ephemeral(String.class);
        
        // Update with value
        channel.update(Collections.singletonList("test"));
        
        // Verify it works
        assertThat(channel.get()).isEqualTo("test");
        
        // Create with key
        EphemeralValue<String> namedChannel = Channels.ephemeral(String.class, "temporary");
        assertThat(namedChannel.getKey()).isEqualTo("temporary");
    }
    
    @Test
    void testNumericOperatorFactories() {
        // Test integer adder
        BinaryOperatorChannel<Integer> intAdder = Channels.integerAdder("int-sum");
        intAdder.update(Arrays.asList(1, 2, 3));
        assertThat(intAdder.get()).isEqualTo(6);
        assertThat(intAdder.getKey()).isEqualTo("int-sum");
        
        // Test long adder
        BinaryOperatorChannel<Long> longAdder = Channels.longAdder("long-sum");
        longAdder.update(Arrays.asList(100L, 200L, 300L));
        assertThat(longAdder.get()).isEqualTo(600L);
        
        // Test double adder
        BinaryOperatorChannel<Double> doubleAdder = Channels.doubleAdder("double-sum");
        doubleAdder.update(Arrays.asList(1.5, 2.5, 3.0));
        assertThat(doubleAdder.get()).isEqualTo(7.0);
        
        // Test integer max
        BinaryOperatorChannel<Integer> intMax = Channels.integerMax("max-value");
        intMax.update(Arrays.asList(5, 10, 3));
        assertThat(intMax.get()).isEqualTo(10);
        
        // Test long max
        BinaryOperatorChannel<Long> longMax = Channels.longMax("long-max");
        longMax.update(Arrays.asList(100L, 500L, 200L));
        assertThat(longMax.get()).isEqualTo(500L);
        
        // Test double max
        BinaryOperatorChannel<Double> doubleMax = Channels.doubleMax("double-max");
        doubleMax.update(Arrays.asList(1.5, 3.5, 2.0));
        assertThat(doubleMax.get()).isEqualTo(3.5);
    }
}