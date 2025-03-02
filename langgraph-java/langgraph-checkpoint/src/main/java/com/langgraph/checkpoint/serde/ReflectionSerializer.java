package com.langgraph.checkpoint.serde;

/**
 * Interface for a serializer that uses reflection to handle arbitrary Java objects.
 */
public interface ReflectionSerializer extends Serializer<Object> {
    /**
     * Register a custom serializer for a specific type.
     *
     * @param type Type to register
     * @param serializer Custom serializer for the type
     * @param <T> Type to register
     */
    <T> void registerSerializer(Class<T> type, TypeSerializer<T> serializer);
    
    /**
     * Register a custom deserializer for a specific type.
     *
     * @param type Type to register
     * @param deserializer Custom deserializer for the type
     * @param <T> Type to register
     */
    <T> void registerDeserializer(Class<T> type, TypeDeserializer<T> deserializer);
}