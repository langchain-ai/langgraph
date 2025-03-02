package com.langgraph.checkpoint.base;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

public class IDTest {

    @Test
    public void testUuidDeterministic() {
        // Same inputs should produce same UUIDs
        UUID uuid1 = ID.uuid("test", "value");
        UUID uuid2 = ID.uuid("test", "value");
        
        assertThat(uuid1).isEqualTo(uuid2);
    }
    
    @Test
    public void testUuidDifferentNamespace() {
        // Different namespaces should produce different UUIDs
        UUID uuid1 = ID.uuid("namespace1", "value");
        UUID uuid2 = ID.uuid("namespace2", "value");
        
        assertThat(uuid1).isNotEqualTo(uuid2);
    }
    
    @Test
    public void testUuidDifferentName() {
        // Different names should produce different UUIDs
        UUID uuid1 = ID.uuid("test", "value1");
        UUID uuid2 = ID.uuid("test", "value2");
        
        assertThat(uuid1).isNotEqualTo(uuid2);
    }
    
    @Test
    public void testCheckpointId() {
        // Checkpoint IDs should be valid UUIDs
        String id = ID.checkpointId("thread-123");
        
        // Should be a valid UUID string
        UUID uuid = UUID.fromString(id);
        assertThat(uuid).isNotNull();
    }
    
    @Test
    public void testUrlSafeId() {
        // URL-safe IDs should be deterministic
        String id1 = ID.urlSafeId("test", "value");
        String id2 = ID.urlSafeId("test", "value");
        
        assertThat(id1).isEqualTo(id2);
        
        // Should not contain padding characters or unsafe URL characters
        assertThat(id1).doesNotContain("=", "+", "/");
    }
}