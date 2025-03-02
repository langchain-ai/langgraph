# Java Serialization Interfaces

This document defines the Java interfaces for the serialization layer of LangGraph, closely aligned with the Python implementation.

## `Serializer` Interface

The base serializer interface providing methods for serializing and deserializing objects.

```java
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
```

## `ReflectionSerializer` Interface

A specialized serializer that can handle arbitrary Java objects by using reflection.

```java
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
```

## `TypeSerializer` and `TypeDeserializer` Interfaces

Interfaces for custom type serialization and deserialization.

```java
package com.langgraph.checkpoint.serde;

/**
 * Interface for serializing a specific type to a format that can be included in MessagePack.
 *
 * @param <T> Type to serialize
 */
public interface TypeSerializer<T> {
    /**
     * Convert object to a serializable representation.
     *
     * @param obj Object to convert
     * @return Serializable representation (must be compatible with MessagePack)
     */
    Object toSerializable(T obj);
}

/**
 * Interface for deserializing a specific type from MessagePack.
 *
 * @param <T> Type to deserialize
 */
public interface TypeDeserializer<T> {
    /**
     * Convert from serialized representation to object.
     *
     * @param serialized Serialized representation
     * @return Deserialized object
     */
    T fromSerialized(Object serialized);
}
```

## `MsgPackSerializer` Implementation

A concrete implementation of `ReflectionSerializer` using MessagePack.

```java
package com.langgraph.checkpoint.serde;

import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;
import org.msgpack.core.MessageUnpacker;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * MessagePack-based serializer that uses reflection to handle arbitrary Java objects.
 * This implementation mirrors the Python serialization approach by saving constructor
 * and import information to reconstruct objects.
 */
public class MsgPackSerializer implements ReflectionSerializer {
    private final Map<Class<?>, TypeSerializer<?>> serializers = new ConcurrentHashMap<>();
    private final Map<Class<?>, TypeDeserializer<?>> deserializers = new ConcurrentHashMap<>();
    private final Map<String, Class<?>> classCache = new ConcurrentHashMap<>();
    
    /**
     * Register built-in serializers for common types.
     */
    public MsgPackSerializer() {
        // Register common built-in types
        registerBuiltinTypes();
    }
    
    private void registerBuiltinTypes() {
        // UUID serializer
        registerSerializer(UUID.class, (uuid) -> uuid.toString());
        registerDeserializer(UUID.class, (str) -> UUID.fromString((String) str));
        
        // Date serializer
        registerSerializer(java.util.Date.class, (date) -> date.getTime());
        registerDeserializer(java.util.Date.class, (millis) -> new Date((Long) millis));
        
        // ... other built-in types as needed
    }
    
    @Override
    public <T> void registerSerializer(Class<T> type, TypeSerializer<T> serializer) {
        serializers.put(type, serializer);
    }
    
    @Override
    public <T> void registerDeserializer(Class<T> type, TypeDeserializer<T> deserializer) {
        deserializers.put(type, deserializer);
    }
    
    @Override
    public byte[] serialize(Object obj) {
        try {
            MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
            serializeObject(obj, packer);
            return packer.toByteArray();
        } catch (IOException e) {
            throw new SerializationException("Failed to serialize object", e);
        }
    }
    
    @Override
    public Object deserialize(byte[] data) {
        try {
            MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(data);
            return deserializeObject(unpacker);
        } catch (IOException e) {
            throw new SerializationException("Failed to deserialize object", e);
        }
    }
    
    /**
     * Serialize an object to the MessagePack packer.
     *
     * @param obj Object to serialize
     * @param packer MessagePack packer
     * @throws IOException If packing fails
     */
    @SuppressWarnings("unchecked")
    private void serializeObject(Object obj, MessageBufferPacker packer) throws IOException {
        if (obj == null) {
            packer.packNil();
            return;
        }
        
        Class<?> type = obj.getClass();
        
        // Check for registered serializer
        if (serializers.containsKey(type)) {
            TypeSerializer<Object> serializer = (TypeSerializer<Object>) serializers.get(type);
            Object serialized = serializer.toSerializable(obj);
            
            // Pack as a special type
            packer.packMapHeader(2);
            packer.packString("__type__");
            packer.packString(type.getName());
            packer.packString("value");
            serializeObject(serialized, packer);
            return;
        }
        
        // Handle primitive types and common objects directly
        if (obj instanceof String) {
            packer.packString((String) obj);
        } else if (obj instanceof Integer) {
            packer.packInt((Integer) obj);
        } else if (obj instanceof Long) {
            packer.packLong((Long) obj);
        } else if (obj instanceof Double) {
            packer.packDouble((Double) obj);
        } else if (obj instanceof Boolean) {
            packer.packBoolean((Boolean) obj);
        } else if (obj instanceof byte[]) {
            packer.packBinaryHeader(((byte[]) obj).length);
            packer.writePayload((byte[]) obj);
        } else if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            packer.packArrayHeader(list.size());
            for (Object item : list) {
                serializeObject(item, packer);
            }
        } else if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            packer.packMapHeader(map.size());
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                serializeObject(entry.getKey(), packer);
                serializeObject(entry.getValue(), packer);
            }
        } else {
            // Custom object - serialize using reflection
            serializeCustomObject(obj, packer);
        }
    }
    
    /**
     * Serialize a custom object using reflection.
     *
     * @param obj Object to serialize
     * @param packer MessagePack packer
     * @throws IOException If packing fails
     */
    private void serializeCustomObject(Object obj, MessageBufferPacker packer) throws IOException {
        Class<?> type = obj.getClass();
        
        // Pack object with type information
        packer.packMapHeader(3);
        packer.packString("__type__");
        packer.packString(type.getName());
        
        // Save constructor info
        packer.packString("__constructor__");
        packer.packString(type.getName());
        
        // Save fields using reflection
        Map<String, Object> fields = getObjectFields(obj);
        packer.packString("__fields__");
        packer.packMapHeader(fields.size());
        
        for (Map.Entry<String, Object> entry : fields.entrySet()) {
            packer.packString(entry.getKey());
            serializeObject(entry.getValue(), packer);
        }
    }
    
    /**
     * Get all fields from an object using reflection.
     *
     * @param obj Object to extract fields from
     * @return Map of field name to field value
     */
    private Map<String, Object> getObjectFields(Object obj) {
        Map<String, Object> result = new HashMap<>();
        Class<?> type = obj.getClass();
        
        // Get all declared fields, including private ones
        for (Field field : type.getDeclaredFields()) {
            try {
                field.setAccessible(true);
                result.put(field.getName(), field.get(obj));
            } catch (IllegalAccessException e) {
                throw new SerializationException("Failed to access field: " + field.getName(), e);
            }
        }
        
        return result;
    }
    
    /**
     * Deserialize an object from the MessagePack unpacker.
     *
     * @param unpacker MessagePack unpacker
     * @return Deserialized object
     * @throws IOException If unpacking fails
     */
    @SuppressWarnings("unchecked")
    private Object deserializeObject(MessageUnpacker unpacker) throws IOException {
        if (unpacker.tryUnpackNil()) {
            return null;
        }
        
        // Handle different types based on MessagePack format
        switch (unpacker.getNextFormat()) {
            case STRING:
                return unpacker.unpackString();
                
            case INTEGER:
                return unpacker.unpackInt();
                
            case FLOAT:
                return unpacker.unpackDouble();
                
            case BOOLEAN:
                return unpacker.unpackBoolean();
                
            case BINARY:
                int binaryLength = unpacker.unpackBinaryHeader();
                byte[] binary = new byte[binaryLength];
                unpacker.readPayload(binary);
                return binary;
                
            case ARRAY:
                int arraySize = unpacker.unpackArrayHeader();
                List<Object> list = new ArrayList<>(arraySize);
                for (int i = 0; i < arraySize; i++) {
                    list.add(deserializeObject(unpacker));
                }
                return list;
                
            case MAP:
                int mapSize = unpacker.unpackMapHeader();
                
                // Check if this is a typed object
                if (mapSize == 2 || mapSize == 3) {
                    String firstKey = unpacker.unpackString();
                    if ("__type__".equals(firstKey)) {
                        String typeName = unpacker.unpackString();
                        String secondKey = unpacker.unpackString();
                        
                        if ("value".equals(secondKey)) {
                            // This is a simple typed value
                            Object value = deserializeObject(unpacker);
                            Class<?> type = loadClass(typeName);
                            
                            if (deserializers.containsKey(type)) {
                                TypeDeserializer<Object> deserializer = 
                                    (TypeDeserializer<Object>) deserializers.get(type);
                                return deserializer.fromSerialized(value);
                            }
                            
                            return value;
                        } else if ("__constructor__".equals(secondKey)) {
                            // This is a complex object with fields
                            String constructorName = unpacker.unpackString();
                            String fieldsKey = unpacker.unpackString();
                            
                            if ("__fields__".equals(fieldsKey)) {
                                int fieldsCount = unpacker.unpackMapHeader();
                                Map<String, Object> fields = new HashMap<>(fieldsCount);
                                
                                for (int i = 0; i < fieldsCount; i++) {
                                    String fieldName = unpacker.unpackString();
                                    Object fieldValue = deserializeObject(unpacker);
                                    fields.put(fieldName, fieldValue);
                                }
                                
                                return reconstructObject(typeName, constructorName, fields);
                            }
                        }
                    }
                }
                
                // Regular map
                Map<Object, Object> map = new HashMap<>(mapSize);
                for (int i = 0; i < mapSize; i++) {
                    Object key = deserializeObject(unpacker);
                    Object value = deserializeObject(unpacker);
                    map.put(key, value);
                }
                return map;
                
            default:
                throw new SerializationException("Unsupported MessagePack format: " + unpacker.getNextFormat());
        }
    }
    
    /**
     * Reconstruct an object using its class name, constructor, and field values.
     *
     * @param typeName Full class name
     * @param constructorName Constructor class name
     * @param fields Map of field names to values
     * @return Reconstructed object
     */
    private Object reconstructObject(String typeName, String constructorName, Map<String, Object> fields) {
        try {
            Class<?> type = loadClass(typeName);
            
            // Try to create instance using no-arg constructor
            Object instance = type.getDeclaredConstructor().newInstance();
            
            // Set all fields using reflection
            for (Map.Entry<String, Object> entry : fields.entrySet()) {
                setField(instance, entry.getKey(), entry.getValue());
            }
            
            return instance;
        } catch (Exception e) {
            throw new SerializationException("Failed to reconstruct object of type: " + typeName, e);
        }
    }
    
    /**
     * Set a field value using reflection.
     *
     * @param obj Object to set field on
     * @param fieldName Field name
     * @param value Field value
     */
    private void setField(Object obj, String fieldName, Object value) {
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (Exception e) {
            throw new SerializationException("Failed to set field: " + fieldName, e);
        }
    }
    
    /**
     * Load a class by name, with caching.
     *
     * @param className Class name to load
     * @return Class object
     */
    private Class<?> loadClass(String className) {
        return classCache.computeIfAbsent(className, name -> {
            try {
                return Class.forName(name);
            } catch (ClassNotFoundException e) {
                throw new SerializationException("Failed to load class: " + name, e);
            }
        });
    }
    
    /**
     * Exception thrown during serialization/deserialization.
     */
    public static class SerializationException extends RuntimeException {
        public SerializationException(String message) {
            super(message);
        }
        
        public SerializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
```

## Record Class Support

Java's Record classes (Java 14+) can be used as a close alternative to Python's TypedDict/Pydantic models for state schemas. The serializer can handle them via reflection.

```java
package com.langgraph.checkpoint.serde;

import java.util.List;
import java.util.Map;

/**
 * Example of a state schema using Java Record (Java 14+).
 * Records provide immutable data classes with automatic getters, 
 * equals/hashCode, and toString implementations.
 */
public record ConversationState(
    List<Map<String, Object>> messages,
    Map<String, Object> context,
    List<String> history
) {
    // Can include custom methods if needed
}

/**
 * Example of how to use Records for state schemas
 */
public class StateExample {
    public static void main(String[] args) {
        // Create a state instance
        ConversationState state = new ConversationState(
            List.of(Map.of("role", "user", "content", "Hello")),
            Map.of("session_id", "12345"),
            List.of("Started conversation")
        );
        
        // Serialize the state
        MsgPackSerializer serializer = new MsgPackSerializer();
        byte[] serialized = serializer.serialize(state);
        
        // Deserialize the state
        ConversationState deserialized = (ConversationState) serializer.deserialize(serialized);
        
        // Access fields using generated getters
        System.out.println(deserialized.messages());
        System.out.println(deserialized.context());
        System.out.println(deserialized.history());
    }
}
```