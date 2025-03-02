package com.langgraph.checkpoint.serde;

/**
 * Interface for serializing and deserializing objects.
 * 
 * @param <T> Type of object to serialize/deserialize
 */
public interface Serializer<T> {
    /**
     * Serialize an object to bytes.
     *
     * @param obj The object to serialize
     * @return Serialized bytes
     */
    byte[] serialize(T obj);
    
    /**
     * Deserialize bytes to an object.
     *
     * @param data The bytes to deserialize
     * @return Deserialized object
     */
    T deserialize(byte[] data);
}