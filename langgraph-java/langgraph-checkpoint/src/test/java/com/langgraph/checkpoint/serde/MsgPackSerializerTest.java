package com.langgraph.checkpoint.serde;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

public class MsgPackSerializerTest {
    
    private MsgPackSerializer serializer;
    
    @BeforeEach
    public void setUp() {
        serializer = new MsgPackSerializer();
    }
    
    @Test
    public void testSerializeDeserializePrimitives() {
        // Test with various primitive types
        assertRoundTrip("Test string");
        assertRoundTrip(123);
        assertRoundTrip(123456789L);
        assertRoundTrip(123.45);
        assertRoundTrip(123.45f);
        assertRoundTrip(true);
        assertRoundTrip(false);
        assertRoundTrip(null);
    }
    
    @Test
    public void testSerializeDeserializeArrays() {
        // Test with arrays and collections
        assertRoundTrip(new byte[] {1, 2, 3, 4, 5});
        assertRoundTrip(Arrays.asList("one", "two", "three"));
        assertRoundTrip(Arrays.asList(1, 2, 3, 4, 5));
    }
    
    @Test
    public void testSerializeDeserializeMaps() {
        // Test with maps
        Map<String, Object> map = new HashMap<>();
        map.put("string", "value");
        map.put("int", 123);
        map.put("boolean", true);
        
        assertRoundTrip(map);
    }
    
    @Test
    public void testSerializeDeserializeNestedStructures() {
        // Test with nested structures
        Map<String, Object> nested = new HashMap<>();
        nested.put("list", Arrays.asList(1, 2, 3));
        nested.put("map", Map.of("key", "value"));
        
        assertRoundTrip(nested);
    }
    
    @Test
    public void testSerializeDeserializeEnums() {
        // Test with enums
        assertRoundTrip(TestEnum.VALUE1);
        assertRoundTrip(TestEnum.VALUE2);
        assertRoundTrip(TestEnum.VALUE3);
    }
    
    @Test
    public void testSerializeDeserializeRecord() {
        // Test with a record
        TestRecord record = new TestRecord("test", 123, Arrays.asList("a", "b", "c"));
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(record);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(TestRecord.class);
        TestRecord deserializedRecord = (TestRecord) deserialized;
        assertThat(deserializedRecord.name()).isEqualTo("test");
        assertThat(deserializedRecord.value()).isEqualTo(123);
        assertThat(deserializedRecord.tags()).containsExactly("a", "b", "c");
    }
    
    @Test
    public void testSerializeDeserializeNestedRecord() {
        // Test with a nested record
        NestedTestRecord record = new NestedTestRecord(
            "parent", 
            new TestRecord("child", 456, Arrays.asList("x", "y", "z"))
        );
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(record);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(NestedTestRecord.class);
        NestedTestRecord deserializedRecord = (NestedTestRecord) deserialized;
        assertThat(deserializedRecord.name()).isEqualTo("parent");
        assertThat(deserializedRecord.child()).isInstanceOf(TestRecord.class);
        assertThat(deserializedRecord.child().name()).isEqualTo("child");
        assertThat(deserializedRecord.child().value()).isEqualTo(456);
        assertThat(deserializedRecord.child().tags()).containsExactly("x", "y", "z");
    }
    
    @Test
    public void testSerializeDeserializeCustomObject() {
        // Test with a custom object
        TestObject obj = new TestObject();
        obj.setName("test");
        obj.setValue(123);
        obj.setActive(true);
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(obj);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(TestObject.class);
        TestObject deserializedObj = (TestObject) deserialized;
        assertThat(deserializedObj.getName()).isEqualTo("test");
        assertThat(deserializedObj.getValue()).isEqualTo(123);
        assertThat(deserializedObj.isActive()).isTrue();
    }
    
    @Test
    public void testSerializeDeserializeInheritance() {
        // Test with inheritance
        ChildTestObject obj = new ChildTestObject();
        obj.setName("parent");
        obj.setValue(123);
        obj.setActive(true);
        obj.setChildProperty("child");
        obj.setChildValue(456);
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(obj);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(ChildTestObject.class);
        ChildTestObject deserializedObj = (ChildTestObject) deserialized;
        assertThat(deserializedObj.getName()).isEqualTo("parent");
        assertThat(deserializedObj.getValue()).isEqualTo(123);
        assertThat(deserializedObj.isActive()).isTrue();
        assertThat(deserializedObj.getChildProperty()).isEqualTo("child");
        assertThat(deserializedObj.getChildValue()).isEqualTo(456);
    }
    
    @Test
    public void testSerializeDeserializeWithCustomSerializer() {
        // Register custom UUID serializer (although built-in one exists)
        serializer.registerSerializer(UUID.class, (uuid) -> uuid.toString().replace("-", ""));
        serializer.registerDeserializer(UUID.class, (str) -> {
            String uuidStr = (String) str;
            // Insert hyphens for standard UUID format
            uuidStr = uuidStr.replaceFirst(
                    "(\\p{XDigit}{8})(\\p{XDigit}{4})(\\p{XDigit}{4})(\\p{XDigit}{4})(\\p{XDigit}+)",
                    "$1-$2-$3-$4-$5");
            return UUID.fromString(uuidStr);
        });
        
        // Test with UUID
        UUID uuid = UUID.randomUUID();
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(uuid);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(UUID.class);
        assertThat(deserialized).isEqualTo(uuid);
    }
    
    @Test
    public void testSerializeDeserializeDateTypes() {
        // Test with Date
        Date date = new Date();
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(date);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(Date.class);
        assertThat(deserialized).isEqualTo(date);
        
        // Test with Java 8 Date/Time types
        Instant instant = Instant.now();
        LocalDate localDate = LocalDate.now();
        LocalTime localTime = LocalTime.now();
        LocalDateTime localDateTime = LocalDateTime.now();
        
        assertRoundTrip(instant);
        assertRoundTrip(localDate);
        assertRoundTrip(localTime);
        assertRoundTrip(localDateTime);
    }
    
    @Test
    public void testTransientFields() {
        // Test with transient fields
        ObjectWithTransient obj = new ObjectWithTransient();
        obj.setPersistent("saved");
        obj.setTransientField("not-saved");
        
        // Serialize and deserialize
        byte[] serialized = serializer.serialize(obj);
        Object deserialized = serializer.deserialize(serialized);
        
        // Verify
        assertThat(deserialized).isInstanceOf(ObjectWithTransient.class);
        ObjectWithTransient deserializedObj = (ObjectWithTransient) deserialized;
        assertThat(deserializedObj.getPersistent()).isEqualTo("saved");
        assertThat(deserializedObj.getTransientField()).isNull(); // Should be null after deserialization
    }
    
    /**
     * Helper method to assert that an object survives a round trip through serialization.
     *
     * @param obj Object to test
     */
    private void assertRoundTrip(Object obj) {
        try {
            // Serialize
            byte[] serialized = serializer.serialize(obj);
            
            // Deserialize
            Object deserialized = serializer.deserialize(serialized);
            
            // Verify
            if (obj instanceof byte[]) {
                // Arrays need special comparison
                assertThat(deserialized).isInstanceOf(byte[].class);
                assertThat((byte[]) deserialized).isEqualTo((byte[]) obj);
            } else if (obj instanceof Number) {
                // For any number type, compare by value instead of exact type
                if (deserialized instanceof Number) {
                    double expected = ((Number) obj).doubleValue();
                    double actual = ((Number) deserialized).doubleValue();
                    assertThat(actual).isCloseTo(expected, within(0.0001));
                } else {
                    throw new AssertionError("Expected Number, got " + 
                        (deserialized != null ? deserialized.getClass().getName() : "null"));
                }
            } else {
                // Special handling for lists
                if (obj instanceof List && deserialized instanceof List) {
                    List<?> originalList = (List<?>) obj;
                    List<?> deserializedList = (List<?>) deserialized;
                    assertThat(deserializedList).hasSameSizeAs(originalList);
                    
                    // Check each element
                    for (int i = 0; i < originalList.size(); i++) {
                        Object originalItem = originalList.get(i);
                        Object deserializedItem = deserializedList.get(i);
                        
                        if (originalItem instanceof Number && deserializedItem instanceof Number) {
                            // Compare numbers by value instead of exact type
                            assertThat(((Number) deserializedItem).doubleValue())
                                .isCloseTo(((Number) originalItem).doubleValue(), within(0.0001));
                        } else {
                            assertThat(deserializedItem).isEqualTo(originalItem);
                        }
                    }
                } else {
                    // Regular equality for other types
                    assertThat(deserialized).isEqualTo(obj);
                }
            }
        } catch (Exception e) {
            throw new AssertionError("Error in roundtrip for " + obj + ": " + e.getMessage(), e);
        }
    }
    
    /**
     * Test enum.
     */
    public enum TestEnum {
        VALUE1, VALUE2, VALUE3
    }
    
    /**
     * Test record class.
     */
    public record TestRecord(String name, int value, List<String> tags) {
    }
    
    /**
     * Nested test record class.
     */
    public record NestedTestRecord(String name, TestRecord child) {
    }
    
    /**
     * Test class for custom object serialization.
     */
    public static class TestObject {
        private String name;
        private int value;
        private boolean active;
        
        public String getName() {
            return name;
        }
        
        public void setName(String name) {
            this.name = name;
        }
        
        public int getValue() {
            return value;
        }
        
        public void setValue(int value) {
            this.value = value;
        }
        
        public boolean isActive() {
            return active;
        }
        
        public void setActive(boolean active) {
            this.active = active;
        }
    }
    
    /**
     * Child test class for inheritance testing.
     */
    public static class ChildTestObject extends TestObject {
        private String childProperty;
        private int childValue;
        
        public String getChildProperty() {
            return childProperty;
        }
        
        public void setChildProperty(String childProperty) {
            this.childProperty = childProperty;
        }
        
        public int getChildValue() {
            return childValue;
        }
        
        public void setChildValue(int childValue) {
            this.childValue = childValue;
        }
    }
    
    /**
     * Test class with transient fields.
     */
    public static class ObjectWithTransient {
        private String persistent;
        private transient String transientField;
        
        public String getPersistent() {
            return persistent;
        }
        
        public void setPersistent(String persistent) {
            this.persistent = persistent;
        }
        
        public String getTransientField() {
            return transientField;
        }
        
        public void setTransientField(String transientField) {
            this.transientField = transientField;
        }
    }
}