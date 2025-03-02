package com.langgraph.checkpoint.serde;

/**
 * Interface for deserializing a specific type from MessagePack.
 *
 * @param <T> Type to deserialize
 */
@FunctionalInterface
public interface TypeDeserializer<T> {
    /**
     * Convert from serialized representation to object.
     *
     * @param serialized Serialized representation
     * @return Deserialized object
     */
    T fromSerialized(Object serialized);
}