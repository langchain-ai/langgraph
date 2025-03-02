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
     * @param valueType The type of values in the channel
     * @param <V> The type of values
     * @return A new LastValue channel
     */
    public static <V> LastValue<V> lastValue(Class<V> valueType) {
        return new LastValue<>(valueType);
    }
    
    /**
     * Creates a LastValue channel with the specified key.
     *
     * @param valueType The type of values in the channel
     * @param key The key (name) of the channel
     * @param <V> The type of values
     * @return A new LastValue channel
     */
    public static <V> LastValue<V> lastValue(Class<V> valueType, String key) {
        return new LastValue<>(valueType, key);
    }
    
    /**
     * Creates a Topic channel.
     *
     * @param valueType The type of values in the channel
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic(Class<V> valueType) {
        return new TopicChannel<>(valueType);
    }
    
    /**
     * Creates a Topic channel with reset-on-consume behavior.
     *
     * @param valueType The type of values in the channel
     * @param resetOnConsume Whether to reset the channel when consumed
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic(Class<V> valueType, boolean resetOnConsume) {
        return new TopicChannel<>(valueType, resetOnConsume);
    }
    
    /**
     * Creates a Topic channel with the specified key.
     *
     * @param valueType The type of values in the channel
     * @param key The key (name) of the channel
     * @param resetOnConsume Whether to reset the channel when consumed
     * @param <V> The type of values
     * @return A new Topic channel
     */
    public static <V> TopicChannel<V> topic(Class<V> valueType, String key, boolean resetOnConsume) {
        return new TopicChannel<>(valueType, key, resetOnConsume);
    }
    
    /**
     * Creates a BinaryOperator channel.
     *
     * @param valueType The type of values in the channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value
     * @param <V> The type of values
     * @return A new BinaryOperator channel
     */
    public static <V> BinaryOperatorChannel<V> binaryOperator(
            Class<V> valueType, BinaryOperator<V> operator, V initialValue) {
        return new BinaryOperatorChannel<>(valueType, operator, initialValue);
    }
    
    /**
     * Creates a BinaryOperator channel with the specified key.
     *
     * @param valueType The type of values in the channel
     * @param key The key (name) of the channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value
     * @param <V> The type of values
     * @return A new BinaryOperator channel
     */
    public static <V> BinaryOperatorChannel<V> binaryOperator(
            Class<V> valueType, String key, BinaryOperator<V> operator, V initialValue) {
        return new BinaryOperatorChannel<>(valueType, key, operator, initialValue);
    }
    
    /**
     * Creates an EphemeralValue channel.
     *
     * @param valueType The type of values in the channel
     * @param <V> The type of values
     * @return A new EphemeralValue channel
     */
    public static <V> EphemeralValue<V> ephemeral(Class<V> valueType) {
        return new EphemeralValue<>(valueType);
    }
    
    /**
     * Creates an EphemeralValue channel with the specified key.
     *
     * @param valueType The type of values in the channel
     * @param key The key (name) of the channel
     * @param <V> The type of values
     * @return A new EphemeralValue channel
     */
    public static <V> EphemeralValue<V> ephemeral(Class<V> valueType, String key) {
        return new EphemeralValue<>(valueType, key);
    }
    
    // Common binary operators for numeric types
    
    /**
     * Creates an Integer adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding integers
     */
    public static BinaryOperatorChannel<Integer> integerAdder(String key) {
        return binaryOperator(Integer.class, key, Integer::sum, 0);
    }
    
    /**
     * Creates a Long adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding longs
     */
    public static BinaryOperatorChannel<Long> longAdder(String key) {
        return binaryOperator(Long.class, key, Long::sum, 0L);
    }
    
    /**
     * Creates a Double adder binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for adding doubles
     */
    public static BinaryOperatorChannel<Double> doubleAdder(String key) {
        return binaryOperator(Double.class, key, Double::sum, 0.0);
    }
    
    /**
     * Creates an Integer max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum integer
     */
    public static BinaryOperatorChannel<Integer> integerMax(String key) {
        return binaryOperator(Integer.class, key, Integer::max, Integer.MIN_VALUE);
    }
    
    /**
     * Creates a Long max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum long
     */
    public static BinaryOperatorChannel<Long> longMax(String key) {
        return binaryOperator(Long.class, key, Long::max, Long.MIN_VALUE);
    }
    
    /**
     * Creates a Double max binary operator channel.
     *
     * @param key The key (name) of the channel
     * @return A new BinaryOperator channel for finding the maximum double
     */
    public static BinaryOperatorChannel<Double> doubleMax(String key) {
        return binaryOperator(Double.class, key, Double::max, Double.MIN_VALUE);
    }
}