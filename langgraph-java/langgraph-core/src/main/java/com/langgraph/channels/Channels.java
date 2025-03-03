package com.langgraph.channels;

import java.util.function.BinaryOperator;

/**
 * Utility class for creating channels easily.
 */
public final class Channels {
    private Channels() {
        // Private constructor to prevent instantiation
    }
    
    /**
     * Creates a LastValue channel.
     *
     * @param <V> The type of values
     * @return A new LastValue channel
     */
    public static <V> LastValue<V> lastValue() {
        return LastValue.<V>create();
    }
    
    /**
     * Creates a LastValue channel with the specified key.
     *
     * @param key The key (name) of the channel
     * @param <V> The type of values
     * @return A new LastValue channel
     */
    public static <V> LastValue<V> lastValue(String key) {
        return LastValue.<V>create(key);
    }
    
    /**
     * Creates a Topic channel.
     *
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic() {
        return TopicChannel.<V>create();
    }
    
    /**
     * Creates a Topic channel with reset-on-consume behavior.
     *
     * @param resetOnConsume Whether to reset the channel when consumed
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic(boolean resetOnConsume) {
        return TopicChannel.<V>create(resetOnConsume);
    }
    
    /**
     * Creates a Topic channel with the specified key.
     *
     * @param key The key (name) of the channel
     * @param resetOnConsume Whether to reset the channel when consumed
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic(String key, boolean resetOnConsume) {
        return TopicChannel.<V>create(key, resetOnConsume);
    }
    
    /**
     * Creates a BinaryOperator channel.
     *
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value
     * @param <V> The type of values
     * @return A new BinaryOperator channel
     */
    public static <V> BinaryOperatorChannel<V> binaryOperator(
            BinaryOperator<V> operator, V initialValue) {
        return BinaryOperatorChannel.<V>create(operator, initialValue);
    }
    
    /**
     * Creates a BinaryOperator channel with the specified key.
     *
     * @param key The key (name) of the channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value
     * @param <V> The type of values
     * @return A new BinaryOperator channel
     */
    public static <V> BinaryOperatorChannel<V> binaryOperator(
            String key, BinaryOperator<V> operator, V initialValue) {
        return BinaryOperatorChannel.<V>create(key, operator, initialValue);
    }
    
    /**
     * Creates an EphemeralValue channel.
     *
     * @param <V> The type of values
     * @return A new EphemeralValue channel
     */
    public static <V> EphemeralValue<V> ephemeral() {
        return EphemeralValue.<V>create();
    }
    
    /**
     * Creates an EphemeralValue channel with the specified key.
     *
     * @param key The key (name) of the channel
     * @param <V> The type of values
     * @return A new EphemeralValue channel
     */
    public static <V> EphemeralValue<V> ephemeral(String key) {
        return EphemeralValue.<V>create(key);
    }
    
    // Common binary operators for numeric types
    
    /**
     * Creates an Integer adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding integers
     */
    public static BinaryOperatorChannel<Integer> integerAdder(String key) {
        return BinaryOperatorChannel.create(key, Integer::sum, 0);
    }
    
    /**
     * Creates a Long adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding longs
     */
    public static BinaryOperatorChannel<Long> longAdder(String key) {
        return BinaryOperatorChannel.create(key, Long::sum, 0L);
    }
    
    /**
     * Creates a Double adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding doubles
     */
    public static BinaryOperatorChannel<Double> doubleAdder(String key) {
        return BinaryOperatorChannel.create(key, Double::sum, 0.0);
    }
    
    /**
     * Creates an Integer max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum integer
     */
    public static BinaryOperatorChannel<Integer> integerMax(String key) {
        return BinaryOperatorChannel.create(key, Integer::max, Integer.MIN_VALUE);
    }
    
    /**
     * Creates a Long max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum long
     */
    public static BinaryOperatorChannel<Long> longMax(String key) {
        return BinaryOperatorChannel.create(key, Long::max, Long.MIN_VALUE);
    }
    
    /**
     * Creates a Double max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum double
     */
    public static BinaryOperatorChannel<Double> doubleMax(String key) {
        return BinaryOperatorChannel.create(key, Double::max, Double.MIN_VALUE);
    }
}