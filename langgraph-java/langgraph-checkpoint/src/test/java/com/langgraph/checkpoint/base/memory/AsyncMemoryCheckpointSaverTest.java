package com.langgraph.checkpoint.base.memory;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

import static org.assertj.core.api.Assertions.assertThat;

public class AsyncMemoryCheckpointSaverTest {
    
    private AsyncMemoryCheckpointSaver saver;
    
    @BeforeEach
    public void setUp() {
        saver = new AsyncMemoryCheckpointSaver();
    }
    
    @Test
    public void testCheckpointAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        Map<String, Object> values = new HashMap<>();
        values.put("key1", "value1");
        values.put("key2", 42);
        
        // Create checkpoint asynchronously
        CompletableFuture<String> future = saver.checkpointAsync(threadId, values);
        
        // Wait for completion
        String checkpointId = future.get();
        
        // Verify checkpoint ID format (should be a UUID)
        assertThat(checkpointId).matches("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$");
        
        // Verify thread has a checkpoint
        CompletableFuture<List<String>> listFuture = saver.listAsync(threadId);
        List<String> checkpoints = listFuture.get();
        assertThat(checkpoints).hasSize(1);
        assertThat(checkpoints).contains(checkpointId);
    }
    
    @Test
    public void testGetValuesAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        Map<String, Object> values = new HashMap<>();
        values.put("key1", "value1");
        values.put("key2", 42);
        
        // Create checkpoint
        String checkpointId = saver.checkpointAsync(threadId, values).get();
        
        // Get values asynchronously
        CompletableFuture<Optional<Map<String, Object>>> future = saver.getValuesAsync(checkpointId);
        Optional<Map<String, Object>> retrievedValues = future.get();
        
        // Verify values
        assertThat(retrievedValues).isPresent();
        assertThat(retrievedValues.get()).containsEntry("key1", "value1");
        assertThat(retrievedValues.get()).containsEntry("key2", 42);
    }
    
    @Test
    public void testListAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        
        // Initially empty
        CompletableFuture<List<String>> initialFuture = saver.listAsync(threadId);
        List<String> initial = initialFuture.get();
        assertThat(initial).isEmpty();
        
        // Create multiple checkpoints
        String id1 = saver.checkpointAsync(threadId, Map.of("key", "value1")).get();
        String id2 = saver.checkpointAsync(threadId, Map.of("key", "value2")).get();
        String id3 = saver.checkpointAsync(threadId, Map.of("key", "value3")).get();
        
        // List checkpoints asynchronously
        CompletableFuture<List<String>> future = saver.listAsync(threadId);
        List<String> checkpoints = future.get();
        
        // Verify order and content
        assertThat(checkpoints).hasSize(3);
        assertThat(checkpoints).containsExactly(id1, id2, id3);
    }
    
    @Test
    public void testLatestAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        
        // Initially empty
        CompletableFuture<Optional<String>> initialFuture = saver.latestAsync(threadId);
        Optional<String> initial = initialFuture.get();
        assertThat(initial).isEmpty();
        
        // Create multiple checkpoints
        saver.checkpointAsync(threadId, Map.of("key", "value1")).get();
        saver.checkpointAsync(threadId, Map.of("key", "value2")).get();
        String id3 = saver.checkpointAsync(threadId, Map.of("key", "value3")).get();
        
        // Get latest asynchronously
        CompletableFuture<Optional<String>> future = saver.latestAsync(threadId);
        Optional<String> latest = future.get();
        
        // Verify latest
        assertThat(latest).isPresent();
        assertThat(latest.get()).isEqualTo(id3);
    }
    
    @Test
    public void testDeleteAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        
        // Create checkpoint
        String checkpointId = saver.checkpointAsync(threadId, Map.of("key", "value")).get();
        
        // Verify checkpoint exists
        assertThat(saver.getValuesAsync(checkpointId).get()).isPresent();
        
        // Delete checkpoint asynchronously
        CompletableFuture<Void> future = saver.deleteAsync(checkpointId);
        future.get(); // Wait for completion
        
        // Verify checkpoint is deleted
        assertThat(saver.getValuesAsync(checkpointId).get()).isEmpty();
    }
    
    @Test
    public void testClearAsync() throws ExecutionException, InterruptedException {
        // Create test data
        String threadId = "test-thread";
        
        // Create multiple checkpoints
        String id1 = saver.checkpointAsync(threadId, Map.of("key", "value1")).get();
        String id2 = saver.checkpointAsync(threadId, Map.of("key", "value2")).get();
        
        // Verify checkpoints exist
        assertThat(saver.listAsync(threadId).get()).hasSize(2);
        
        // Clear thread asynchronously
        CompletableFuture<Void> future = saver.clearAsync(threadId);
        future.get(); // Wait for completion
        
        // Verify checkpoints are deleted
        assertThat(saver.listAsync(threadId).get()).isEmpty();
    }
}