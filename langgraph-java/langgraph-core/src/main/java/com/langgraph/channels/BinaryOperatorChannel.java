package com.langgraph.channels;

import java.util.List;
import java.util.function.BinaryOperator;

/**
 * A channel that aggregates values using a binary operator.
 * This is useful for operations like sum, max, min, etc.
 *
 * @param <V> Type of the value stored in the channel
 */
public class BinaryOperatorChannel<V> extends AbstractChannel<V, V, V> {
    /**
     * The binary operator to apply for aggregation.
     */
    private final BinaryOperator<V> operator;
    
    /**
     * The current value, null if the channel has not been updated yet.
     */
    private V value;
    
    /**
     * The initial value to use if none has been set yet.
     */
    private final V initialValue;
    
    /**
     * Flag to track if this channel has been initialized.
     */
    private boolean initialized = false;
    
    /**
     * Creates a new BinaryOperatorChannel with the specified value type and operator.
     *
     * @param typeRef The TypeReference for the value type
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value to use if none has been set yet
     */
    protected BinaryOperatorChannel(TypeReference<V> typeRef, BinaryOperator<V> operator, V initialValue) {
        super(typeRef, typeRef, typeRef); // For BinaryOperatorChannel, V=U=C
        this.operator = operator;
        this.initialValue = initialValue;
    }
    
    /**
     * Creates a new BinaryOperatorChannel with the specified value type, key, and operator.
     *
     * @param typeRef The TypeReference for the value type
     * @param key The key (name) of this channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value to use if none has been set yet
     */
    protected BinaryOperatorChannel(TypeReference<V> typeRef, String key, BinaryOperator<V> operator, V initialValue) {
        super(typeRef, typeRef, typeRef, key); // For BinaryOperatorChannel, V=U=C
        this.operator = operator;
        this.initialValue = initialValue;
    }
    
    /**
     * Factory method to create a BinaryOperatorChannel with proper generic type capture.
     * 
     * <p>Example usage:
     * <pre>
     * BinaryOperatorChannel&lt;Integer&gt; channel = BinaryOperatorChannel.&lt;Integer&gt;create(Integer::sum, 0);
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value to use if none has been set yet
     * @return A new BinaryOperatorChannel with the captured type parameter
     */
    public static <T> BinaryOperatorChannel<T> create(BinaryOperator<T> operator, T initialValue) {
        return new BinaryOperatorChannel<>(new TypeReference<T>() {}, operator, initialValue);
    }
    
    /**
     * Factory method to create a BinaryOperatorChannel with proper generic type capture
     * and a specified key.
     * 
     * <p>Example usage:
     * <pre>
     * BinaryOperatorChannel&lt;Integer&gt; channel = BinaryOperatorChannel.&lt;Integer&gt;create("counter", Integer::sum, 0);
     * </pre>
     * 
     * @param <T> The type parameter for the channel
     * @param key The key (name) for the channel
     * @param operator The binary operator to use for aggregation
     * @param initialValue The initial value to use if none has been set yet
     * @return A new BinaryOperatorChannel with the captured type parameter and specified key
     */
    public static <T> BinaryOperatorChannel<T> create(String key, BinaryOperator<T> operator, T initialValue) {
        return new BinaryOperatorChannel<>(new TypeReference<T>() {}, key, operator, initialValue);
    }
    
    @Override
    public boolean update(List<V> values) {
        if (values.isEmpty()) {
            return false;
        }
        
        V current = initialized ? this.value : initialValue;
        
        for (V val : values) {
            current = operator.apply(current, val);
        }
        
        this.value = current;
        initialized = true;
        return true;
    }
    
    @Override
    public V get() throws EmptyChannelException {
        if (!initialized) {
            throw new EmptyChannelException(
                "BinaryOperatorChannel at key '" + key + "' is empty (never updated)");
        }
        return value;
    }
    
    @Override
    public BaseChannel<V, V, V> fromCheckpoint(V checkpoint) {
        BinaryOperatorChannel<V> newChannel = new BinaryOperatorChannel<>(
            valueTypeRef, key, operator, initialValue);
        // Even null is a valid checkpoint value - it means the channel was initialized with null
        newChannel.value = checkpoint;
        newChannel.initialized = true;
        return newChannel;
    }
    
    /**
     * Returns the string representation of this channel.
     *
     * @return String representation
     */
    @Override
    public String toString() {
        return "BinaryOperator(" + (initialized ? value : "empty") + ")";
    }
    
    /**
     * Returns the binary operator.
     *
     * @return The binary operator
     */
    public BinaryOperator<V> getOperator() {
        return operator;
    }
    
    /**
     * Returns the initial value.
     *
     * @return The initial value
     */
    public V getInitialValue() {
        return initialValue;
    }
}