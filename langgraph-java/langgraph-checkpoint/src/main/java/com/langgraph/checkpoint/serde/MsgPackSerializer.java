package com.langgraph.checkpoint.serde;

import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;
import org.msgpack.core.MessageUnpacker;
import org.msgpack.core.MessageFormat;

import java.io.IOException;
import java.lang.reflect.*;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * MessagePack-based serializer that uses reflection to handle arbitrary Java objects.
 * Supports primitive types, collections, maps, records, and custom objects.
 */
public class MsgPackSerializer implements ReflectionSerializer {
    private final Map<Class<?>, TypeSerializer<?>> serializers = new ConcurrentHashMap<>();
    private final Map<Class<?>, TypeDeserializer<?>> deserializers = new ConcurrentHashMap<>();
    private final Map<Class<?>, RecordInfo> recordInfoCache = new ConcurrentHashMap<>();
    
    /**
     * Record component information cache to avoid repeated reflection.
     */
    private static class RecordInfo {
        final RecordComponent[] components;
        final Constructor<?> constructor;
        
        RecordInfo(RecordComponent[] components, Constructor<?> constructor) {
            this.components = components;
            this.constructor = constructor;
        }
    }
    
    /**
     * Register built-in serializers for common types.
     */
    public MsgPackSerializer() {
        registerBuiltinTypes();
    }
    
    /**
     * Register built-in serializers for common types.
     */
    private void registerBuiltinTypes() {
        // UUID serializer
        registerSerializer(UUID.class, (uuid) -> uuid.toString());
        registerDeserializer(UUID.class, (str) -> UUID.fromString((String) str));
        
        // Date serializer
        registerSerializer(java.util.Date.class, (date) -> date.getTime());
        registerDeserializer(java.util.Date.class, (millis) -> new Date((Long) millis));
        
        // Java 8 Date/Time API
        registerSerializer(Instant.class, (instant) -> instant.toString());
        registerDeserializer(Instant.class, (str) -> Instant.parse((String) str));
        
        registerSerializer(LocalDate.class, (date) -> date.toString());
        registerDeserializer(LocalDate.class, (str) -> LocalDate.parse((String) str));
        
        registerSerializer(LocalTime.class, (time) -> time.toString());
        registerDeserializer(LocalTime.class, (str) -> LocalTime.parse((String) str));
        
        registerSerializer(LocalDateTime.class, (dateTime) -> dateTime.toString());
        registerDeserializer(LocalDateTime.class, (str) -> LocalDateTime.parse((String) str));
        
        // Add more built-in serializers as needed
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
        } else if (obj instanceof Float) {
            packer.packFloat((Float) obj);
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
        } else if (obj instanceof Enum<?>) {
            // Handle enums by name
            packer.packMapHeader(2);
            packer.packString("__type__");
            packer.packString(type.getName());
            packer.packString("value");
            packer.packString(((Enum<?>) obj).name());
        } else if (type.isRecord()) {
            // Handle Record types
            serializeRecord(obj, packer);
        } else {
            // Handle custom objects with reflection
            serializeCustomObject(obj, packer);
        }
    }
    
    /**
     * Serialize a Record object.
     *
     * @param record The record to serialize
     * @param packer The MessagePack packer
     * @throws IOException If packing fails
     */
    private void serializeRecord(Object record, MessageBufferPacker packer) throws IOException {
        Class<?> recordClass = record.getClass();
        
        // Pack as a special type with fields
        packer.packMapHeader(2);
        packer.packString("__type__");
        packer.packString(recordClass.getName());
        packer.packString("fields");
        
        RecordComponent[] components = recordClass.getRecordComponents();
        packer.packMapHeader(components.length);
        
        for (RecordComponent component : components) {
            packer.packString(component.getName());
            try {
                Method accessor = component.getAccessor();
                Object value = accessor.invoke(record);
                serializeObject(value, packer);
            } catch (ReflectiveOperationException e) {
                throw new SerializationException("Failed to access record component: " + component.getName(), e);
            }
        }
    }
    
    /**
     * Serialize a custom object using reflection.
     *
     * @param obj The object to serialize
     * @param packer The MessagePack packer
     * @throws IOException If packing fails
     */
    private void serializeCustomObject(Object obj, MessageBufferPacker packer) throws IOException {
        Class<?> objClass = obj.getClass();
        
        // Pack as a special type with fields
        packer.packMapHeader(2);
        packer.packString("__type__");
        packer.packString(objClass.getName());
        packer.packString("fields");
        
        // Get all fields including inherited ones
        List<Field> fields = getAllFields(objClass);
        
        // Filter out transient fields
        List<Field> serializableFields = fields.stream()
            .filter(field -> !Modifier.isTransient(field.getModifiers()) && 
                            !Modifier.isStatic(field.getModifiers()))
            .toList();
        
        packer.packMapHeader(serializableFields.size());
        
        for (Field field : serializableFields) {
            packer.packString(field.getName());
            try {
                field.setAccessible(true);
                Object value = field.get(obj);
                serializeObject(value, packer);
            } catch (IllegalAccessException e) {
                throw new SerializationException("Failed to access field: " + field.getName(), e);
            }
        }
    }
    
    /**
     * Get all fields for a class including inherited fields.
     *
     * @param clazz The class to get fields for
     * @return List of all fields
     */
    private List<Field> getAllFields(Class<?> clazz) {
        List<Field> fields = new ArrayList<>();
        Class<?> currentClass = clazz;
        
        while (currentClass != null && currentClass != Object.class) {
            fields.addAll(Arrays.asList(currentClass.getDeclaredFields()));
            currentClass = currentClass.getSuperclass();
        }
        
        return fields;
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
        if (!unpacker.hasNext()) {
            throw new SerializationException("Unexpected end of data");
        }
        
        if (unpacker.tryUnpackNil()) {
            return null;
        }
        
        MessageFormat format = unpacker.getNextFormat();
        
        if (format == MessageFormat.STR8 || 
            format == MessageFormat.STR16 || 
            format == MessageFormat.STR32 || 
            format == MessageFormat.FIXSTR) {
            return unpacker.unpackString();
        } else if (format == MessageFormat.INT8 || 
                  format == MessageFormat.INT16 || 
                  format == MessageFormat.INT32 || 
                  format == MessageFormat.INT64 || 
                  format == MessageFormat.UINT8 || 
                  format == MessageFormat.UINT16 || 
                  format == MessageFormat.UINT32 || 
                  format == MessageFormat.UINT64 || 
                  format == MessageFormat.POSFIXINT || 
                  format == MessageFormat.NEGFIXINT) {
            if (format == MessageFormat.INT64 || format == MessageFormat.UINT64) {
                return unpacker.unpackLong();
            } else {
                try {
                    return unpacker.unpackInt();
                } catch (Exception e) {
                    // Fallback to long if int unpacking fails
                    return unpacker.unpackLong();
                }
            }
        } else if (format == MessageFormat.FLOAT32 || 
                  format == MessageFormat.FLOAT64) {
            return unpacker.unpackDouble();
        } else if (format == MessageFormat.BOOLEAN) {
            return unpacker.unpackBoolean();
        } else if (format == MessageFormat.BIN8 || 
                  format == MessageFormat.BIN16 || 
                  format == MessageFormat.BIN32) {
            int binaryLength = unpacker.unpackBinaryHeader();
            byte[] binary = new byte[binaryLength];
            unpacker.readPayload(binary);
            return binary;
        } else if (format == MessageFormat.ARRAY16 || 
                  format == MessageFormat.ARRAY32 || 
                  format == MessageFormat.FIXARRAY) {
            int arraySize = unpacker.unpackArrayHeader();
            List<Object> list = new ArrayList<>(arraySize);
            for (int i = 0; i < arraySize; i++) {
                list.add(deserializeObject(unpacker));
            }
            return list;
        } else if (format == MessageFormat.MAP16 || 
                  format == MessageFormat.MAP32 || 
                  format == MessageFormat.FIXMAP) {
            int mapSize = unpacker.unpackMapHeader();
            
            // Handle empty map
            if (mapSize == 0) {
                return new HashMap<>();
            }
            
            // Check for special type marker
            Object firstKey = deserializeObject(unpacker);
            if (mapSize == 2 && firstKey instanceof String && "__type__".equals(firstKey)) {
                String typeName = (String) deserializeObject(unpacker);
                
                // Get the second key
                Object secondKey = deserializeObject(unpacker);
                
                if (secondKey instanceof String) {
                    String secondKeyStr = (String) secondKey;
                    
                    try {
                        Class<?> type = Class.forName(typeName);
                        
                        // Check for registered deserializer
                        if ("value".equals(secondKeyStr) && deserializers.containsKey(type)) {
                            Object serialized = deserializeObject(unpacker);
                            TypeDeserializer<Object> deserializer = 
                                (TypeDeserializer<Object>) deserializers.get(type);
                            return deserializer.fromSerialized(serialized);
                        }
                        
                        // Handle enums
                        if ("value".equals(secondKeyStr) && type.isEnum()) {
                            String enumValue = (String) deserializeObject(unpacker);
                            return Enum.valueOf((Class<Enum>) type, enumValue);
                        }
                        
                        // Handle records
                        if ("fields".equals(secondKeyStr) && type.isRecord()) {
                            return deserializeRecord(type, unpacker);
                        }
                        
                        // Handle custom objects
                        if ("fields".equals(secondKeyStr)) {
                            return deserializeCustomObject(type, unpacker);
                        }
                    } catch (ClassNotFoundException e) {
                        // If class not found, fall back to regular map deserialization
                    } catch (ReflectiveOperationException e) {
                        throw new SerializationException("Failed to deserialize object of type " + typeName, e);
                    }
                    
                    // If special type handling failed, read the value to keep unpacker consistent
                    Object secondValue = deserializeObject(unpacker);
                    
                    // Create a fallback map with the special type info
                    Map<Object, Object> fallbackMap = new HashMap<>();
                    fallbackMap.put(firstKey, typeName);
                    fallbackMap.put(secondKey, secondValue);
                    return fallbackMap;
                }
                
                // If the second key wasn't a string as expected, we need to handle it as a regular map
                Object firstValue = deserializeObject(unpacker);
                
                // Create a map with the first key-value pair
                Map<Object, Object> map = new HashMap<>(mapSize);
                map.put(firstKey, firstValue);
                
                // Read the remaining entries
                for (int i = 1; i < mapSize; i++) {
                    Object key = deserializeObject(unpacker);
                    Object value = deserializeObject(unpacker);
                    map.put(key, value);
                }
                
                return map;
            } else {
                // Regular map - we already read the first key
                Map<Object, Object> map = new HashMap<>(mapSize);
                
                // Read the first value
                Object firstValue = deserializeObject(unpacker);
                map.put(firstKey, firstValue);
                
                // Read the remaining entries
                for (int i = 1; i < mapSize; i++) {
                    Object key = deserializeObject(unpacker);
                    Object value = deserializeObject(unpacker);
                    map.put(key, value);
                }
                
                return map;
            }
        }
        
        // Default case
        throw new SerializationException("Unsupported MessagePack format: " + format);
    }
    
    /**
     * Deserialize a record.
     *
     * @param recordClass The record class
     * @param unpacker The unpacker containing the fields map
     * @return The deserialized record
     * @throws IOException If unpacking fails
     * @throws ReflectiveOperationException If reflection operations fail
     */
    private Object deserializeRecord(Class<?> recordClass, MessageUnpacker unpacker) 
            throws IOException, ReflectiveOperationException {
        
        // Get record info from cache or create it
        RecordInfo recordInfo = recordInfoCache.computeIfAbsent(recordClass, cls -> {
            try {
                RecordComponent[] components = cls.getRecordComponents();
                Class<?>[] paramTypes = Arrays.stream(components)
                    .map(RecordComponent::getType)
                    .toArray(Class<?>[]::new);
                Constructor<?> constructor = cls.getDeclaredConstructor(paramTypes);
                constructor.setAccessible(true);
                return new RecordInfo(components, constructor);
            } catch (NoSuchMethodException e) {
                throw new SerializationException("Failed to get constructor for record: " + cls.getName(), e);
            }
        });
        
        // Read the fields map
        int fieldCount = unpacker.unpackMapHeader();
        Map<String, Object> fieldValues = new HashMap<>(fieldCount);
        
        for (int i = 0; i < fieldCount; i++) {
            String fieldName = (String) deserializeObject(unpacker);
            Object fieldValue = deserializeObject(unpacker);
            fieldValues.put(fieldName, fieldValue);
        }
        
        // Prepare constructor arguments in the correct order
        Object[] constructorArgs = new Object[recordInfo.components.length];
        for (int i = 0; i < recordInfo.components.length; i++) {
            RecordComponent component = recordInfo.components[i];
            Object value = fieldValues.get(component.getName());
            constructorArgs[i] = value;
        }
        
        // Create the record instance
        return recordInfo.constructor.newInstance(constructorArgs);
    }
    
    /**
     * Deserialize a custom object.
     *
     * @param objectClass The object class
     * @param unpacker The unpacker containing the fields map
     * @return The deserialized object
     * @throws IOException If unpacking fails
     * @throws ReflectiveOperationException If reflection operations fail
     */
    private Object deserializeCustomObject(Class<?> objectClass, MessageUnpacker unpacker) 
            throws IOException, ReflectiveOperationException {
        
        // Create instance using default constructor
        Constructor<?> constructor;
        try {
            constructor = objectClass.getDeclaredConstructor();
            constructor.setAccessible(true);
        } catch (NoSuchMethodException e) {
            throw new SerializationException(
                "Class " + objectClass.getName() + " must have a no-arg constructor for deserialization", e);
        }
        
        Object instance = constructor.newInstance();
        
        // Read the fields map
        int fieldCount = unpacker.unpackMapHeader();
        
        for (int i = 0; i < fieldCount; i++) {
            String fieldName = (String) deserializeObject(unpacker);
            Object fieldValue = deserializeObject(unpacker);
            
            try {
                // Find the field (including in superclasses)
                Field field = findField(objectClass, fieldName);
                if (field != null) {
                    field.setAccessible(true);
                    field.set(instance, fieldValue);
                }
            } catch (NoSuchFieldException e) {
                // Skip fields that don't exist in the current class version
            }
        }
        
        return instance;
    }
    
    /**
     * Find a field in a class or its superclasses.
     *
     * @param clazz The class to search
     * @param fieldName The field name to find
     * @return The found field
     * @throws NoSuchFieldException If the field is not found
     */
    private Field findField(Class<?> clazz, String fieldName) throws NoSuchFieldException {
        Class<?> currentClass = clazz;
        while (currentClass != null) {
            try {
                return currentClass.getDeclaredField(fieldName);
            } catch (NoSuchFieldException e) {
                currentClass = currentClass.getSuperclass();
            }
        }
        throw new NoSuchFieldException("Field not found: " + fieldName);
    }
}