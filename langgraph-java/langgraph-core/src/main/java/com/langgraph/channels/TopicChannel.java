package com.langgraph.channels;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A channel that collects values into a list.
 * Unlike LastValue, it can receive multiple values per update.
 *
 * @param <V> Type of the value stored in the channel
 */
public class TopicChannel<V> extends AbstractChannel<List<V>, V, List<V>> {
    /**
     * The current list of values, empty if the channel has not been updated yet.
     */
    private List<V> values = new ArrayList<>();
    
    /**
     * Flag to track if this channel has been initialized.
     */
    private boolean initialized = false;
    
    /**
     * Flag to determine if the channel should reset after consumption.
     */
    private final boolean resetOnConsume;
    
    /**
     * Creates a new Topic channel using TypeReference with specified reset behavior.
     *
     * @param elementTypeRef TypeReference capturing the element type
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    protected TopicChannel(TypeReference<V> elementTypeRef, boolean resetOnConsume) {
        // Create TypeReferences for the other types (List<V> in this case)
        super(
            createListTypeReference(elementTypeRef),  // Value type (List<V>)
            elementTypeRef,                          // Update type (V)
            createListTypeReference(elementTypeRef)   // Checkpoint type (List<V>)
        );
        this.resetOnConsume = resetOnConsume;
    }
    
    /**
     * Creates a new Topic channel using TypeReference with specified key and reset behavior.
     *
     * @param elementTypeRef TypeReference capturing the element type
     * @param key The key (name) of this channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     */
    protected TopicChannel(TypeReference<V> elementTypeRef, String key, boolean resetOnConsume) {
        // Create TypeReferences for the other types (List<V> in this case)
        super(
            createListTypeReference(elementTypeRef),  // Value type (List<V>)
            elementTypeRef,                          // Update type (V)
            createListTypeReference(elementTypeRef),  // Checkpoint type (List<V>)
            key
        );
        this.resetOnConsume = resetOnConsume;
    }
    
    /**
     * Factory method to create a TypeReference for List<V> given a TypeReference for V.
     *
     * @param <V> The element type
     * @param elementTypeRef The TypeReference for the element type
     * @return A TypeReference for List<V>
     */
    @SuppressWarnings("unchecked")
    private static <V> TypeReference<List<V>> createListTypeReference(final TypeReference<V> elementTypeRef) {
        final Type elementType = elementTypeRef.getType();
        
        return new TypeReference<List<V>>() {
            @Override
            public Type getType() {
                // Create a ParameterizedType for List<V>
                return new ParameterizedType() {
                    @Override
                    public Type[] getActualTypeArguments() {
                        return new Type[] { elementType };
                    }
                    
                    @Override
                    public Type getRawType() {
                        return List.class;
                    }
                    
                    @Override
                    public Type getOwnerType() {
                        return null;
                    }
                    
                    @Override
                    public String toString() {
                        return "java.util.List<" + elementType + ">";
                    }
                };
            }
            
            @Override
            public Class<List<V>> getRawClass() {
                return (Class<List<V>>) (Class<?>) List.class;
            }
        };
    }
    
    /**
     * Factory method to create a TopicChannel with proper generic type inference.
     * 
     * <p>Example usage:
     * <pre>
     * TopicChannel&lt;Integer&gt; channel = TopicChannel.&lt;Integer&gt;create();
     * </pre>
     * 
     * @param <T> The element type parameter for the channel
     * @return A new TopicChannel with the captured type parameter
     */
    public static <T> TopicChannel<T> create() {
        return new TopicChannel<>(new TypeReference<T>() {}, false);
    }
    
    /**
     * Factory method to create a TopicChannel with proper generic type inference
     * and a specified key.
     * 
     * <p>Example usage:
     * <pre>
     * TopicChannel&lt;Integer&gt; channel = TopicChannel.&lt;Integer&gt;create("myChannel");
     * </pre>
     * 
     * @param <T> The element type parameter for the channel
     * @param key The key (name) for the channel
     * @return A new TopicChannel with the captured type parameter and specified key
     */
    public static <T> TopicChannel<T> create(String key) {
        return new TopicChannel<>(new TypeReference<T>() {}, key, false);
    }
    
    /**
     * Factory method to create a TopicChannel with proper generic type inference,
     * specified key, and reset behavior.
     * 
     * <p>Example usage:
     * <pre>
     * TopicChannel&lt;Integer&gt; channel = TopicChannel.&lt;Integer&gt;create(true);
     * TopicChannel&lt;String&gt; channel = TopicChannel.&lt;String&gt;create("myChannel", true);
     * </pre>
     * 
     * @param <T> The element type parameter for the channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     * @return A new TopicChannel with the captured type parameter
     */
    public static <T> TopicChannel<T> create(boolean resetOnConsume) {
        return new TopicChannel<>(new TypeReference<T>() {}, resetOnConsume);
    }
    
    /**
     * Factory method to create a TopicChannel with proper generic type inference,
     * specified key, and reset behavior.
     * 
     * @param <T> The element type parameter for the channel
     * @param key The key (name) for the channel
     * @param resetOnConsume Whether to reset the channel after consume() is called
     * @return A new TopicChannel with the captured type parameter and specified settings
     */
    public static <T> TopicChannel<T> create(String key, boolean resetOnConsume) {
        return new TopicChannel<>(new TypeReference<T>() {}, key, resetOnConsume);
    }
    
    @Override
    public boolean update(List<V> newValues) {
        if (newValues.isEmpty()) {
            return false;
        }
        
        values.addAll(newValues);
        initialized = true;
        return true;
    }
    
    /**
     * Updates the channel with a single new value.
     * This is a convenience method for handling cases where the update comes as a single value
     * instead of a list.
     *
     * @param newValue The new value to add to the topic
     * @return true if the channel was updated, false otherwise
     */
    @Override
    public boolean updateSingleValue(V newValue) {
        if (newValue == null) {
            return false;
        }
        
        values.add(newValue);
        initialized = true;
        return true;
    }
    
    @Override
    public List<V> get() throws EmptyChannelException {
        // Always return the current list (empty or not) for Python compatibility
        // This prevents EmptyChannelException when accessing uninitialized channels
        return Collections.unmodifiableList(values);
    }
    
    @Override
    public BaseChannel<List<V>, V, List<V>> fromCheckpoint(List<V> checkpoint) {
        // Get the element type reference from the updateTypeRef
        TypeReference<V> elementTypeRef = updateTypeRef;
        
        // Create a new channel with the same type information
        TopicChannel<V> newChannel = new TopicChannel<>(elementTypeRef, key, resetOnConsume);
        
        // Restore the values from checkpoint
        if (checkpoint != null) {
            newChannel.values = new ArrayList<>(checkpoint);
            newChannel.initialized = true;
        }
        
        return newChannel;
    }
    
    @Override
    public boolean consume() {
        if (resetOnConsume && initialized) {
            values.clear();
            initialized = false;
            return true;
        }
        return false;
    }
    
    /**
     * Returns the element type class.
     *
     * @return The element type class
     */
    public Class<V> getElementType() {
        return updateTypeRef.getRawClass();
    }
    
    /**
     * Returns the string representation of this channel.
     *
     * @return String representation
     */
    @Override
    public String toString() {
        return "Topic(" + (initialized ? values : "empty") + ")";
    }
    
    /**
     * Checks if this channel is equal to another object.
     *
     * @param obj The object to compare with
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof TopicChannel)) {
            return false;
        }
        if (!super.equals(obj)) {
            return false;
        }
        
        TopicChannel<?> other = (TopicChannel<?>) obj;
        return initialized == other.initialized &&
               resetOnConsume == other.resetOnConsume &&
               values.equals(other.values);
    }
    
    /**
     * Returns the hash code of this channel.
     *
     * @return The hash code
     */
    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (initialized ? 1 : 0);
        result = 31 * result + (resetOnConsume ? 1 : 0);
        result = 31 * result + values.hashCode();
        return result;
    }
}