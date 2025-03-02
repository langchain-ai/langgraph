package com.langgraph.checkpoint.serde;

/**
 * Interface for serializing a specific type to a format that can be included in MessagePack.
 *
 * @param <T> Type to serialize
 */
@FunctionalInterface
public interface TypeSerializer<T> {
    /**
     * Convert object to a serializable representation.
     *
     * @param obj Object to convert
     * @return Serializable representation (must be compatible with MessagePack)
     */
    Object toSerializable(T obj);
}