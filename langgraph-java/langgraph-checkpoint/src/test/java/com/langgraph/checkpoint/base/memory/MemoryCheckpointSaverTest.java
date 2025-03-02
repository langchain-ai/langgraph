package com.langgraph.checkpoint.base.memory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

public class MemoryCheckpointSaverTest {
    
    private MemoryCheckpointSaver saver;
    
    @BeforeEach
    public void setUp() {
        saver = new MemoryCheckpointSaver();
    }
    
    @Test
    public void testCheckpoint() {
        // Create test data
        String threadId = "test-thread";
        Map<String, Object> values = new HashMap<>();
        values.put("key1", "value1");
        values.put("key2", 42);
        
        // Create checkpoint
        String checkpointId = saver.checkpoint(threadId, values);
        
        // Verify checkpoint ID format (should be a UUID)
        assertThat(checkpointId).matches("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$");
        
        // Verify thread has a checkpoint
        List<String> checkpoints = saver.list(threadId);
        assertThat(checkpoints).hasSize(1);
        assertThat(checkpoints).contains(checkpointId);
        
        // Verify latest checkpoint
        Optional<String> latest = saver.latest(threadId);
        assertThat(latest).isPresent();
        assertThat(latest.get()).isEqualTo(checkpointId);
    }
    
    @Test
    public void testGetValues() {
        // Create test data
        String threadId = "test-thread";
        Map<String, Object> values = new HashMap<>();
        values.put("key1", "value1");
        values.put("key2", 42);
        
        // Create checkpoint
        String checkpointId = saver.checkpoint(threadId, values);
        
        // Get values
        Optional<Map<String, Object>> retrievedValues = saver.getValues(checkpointId);
        
        // Verify values
        assertThat(retrievedValues).isPresent();
        assertThat(retrievedValues.get()).containsEntry("key1", "value1");
        assertThat(retrievedValues.get()).containsEntry("key2", 42);
        
        // Verify non-existent checkpoint
        Optional<Map<String, Object>> nonExistent = saver.getValues("non-existent");
        assertThat(nonExistent).isEmpty();
    }
    
    @Test
    public void testList() {
        // Create test data
        String threadId = "test-thread";
        
        // Initially empty
        List<String> initial = saver.list(threadId);
        assertThat(initial).isEmpty();
        
        // Create multiple checkpoints
        String id1 = saver.checkpoint(threadId, Map.of("key", "value1"));
        String id2 = saver.checkpoint(threadId, Map.of("key", "value2"));
        String id3 = saver.checkpoint(threadId, Map.of("key", "value3"));
        
        // List checkpoints
        List<String> checkpoints = saver.list(threadId);
        
        // Verify order and content
        assertThat(checkpoints).hasSize(3);
        assertThat(checkpoints).containsExactly(id1, id2, id3);
        
        // Different thread should have no checkpoints
        List<String> otherThread = saver.list("other-thread");
        assertThat(otherThread).isEmpty();
    }
    
    @Test
    public void testLatest() {
        // Create test data
        String threadId = "test-thread";
        
        // Initially empty
        Optional<String> initial = saver.latest(threadId);
        assertThat(initial).isEmpty();
        
        // Create multiple checkpoints
        saver.checkpoint(threadId, Map.of("key", "value1"));
        saver.checkpoint(threadId, Map.of("key", "value2"));
        String id3 = saver.checkpoint(threadId, Map.of("key", "value3"));
        
        // Get latest
        Optional<String> latest = saver.latest(threadId);
        
        // Verify latest
        assertThat(latest).isPresent();
        assertThat(latest.get()).isEqualTo(id3);
    }
    
    @Test
    public void testDelete() {
        // Create test data
        String threadId = "test-thread";
        
        // Create checkpoint
        String checkpointId = saver.checkpoint(threadId, Map.of("key", "value"));
        
        // Verify checkpoint exists
        assertThat(saver.getValues(checkpointId)).isPresent();
        assertThat(saver.list(threadId)).contains(checkpointId);
        
        // Delete checkpoint
        saver.delete(checkpointId);
        
        // Verify checkpoint is deleted
        assertThat(saver.getValues(checkpointId)).isEmpty();
        assertThat(saver.list(threadId)).doesNotContain(checkpointId);
    }
    
    @Test
    public void testClear() {
        // Create test data
        String threadId = "test-thread";
        
        // Create multiple checkpoints
        String id1 = saver.checkpoint(threadId, Map.of("key", "value1"));
        String id2 = saver.checkpoint(threadId, Map.of("key", "value2"));
        
        // Verify checkpoints exist
        assertThat(saver.list(threadId)).hasSize(2);
        assertThat(saver.getValues(id1)).isPresent();
        assertThat(saver.getValues(id2)).isPresent();
        
        // Clear thread
        saver.clear(threadId);
        
        // Verify checkpoints are deleted
        assertThat(saver.list(threadId)).isEmpty();
        assertThat(saver.getValues(id1)).isEmpty();
        assertThat(saver.getValues(id2)).isEmpty();
    }
}