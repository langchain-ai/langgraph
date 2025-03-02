package com.langgraph.checkpoint.base.memory;

import com.langgraph.checkpoint.base.AsyncBaseCheckpointSaver;
import com.langgraph.checkpoint.base.BaseCheckpointSaver;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Asynchronous in-memory implementation of a checkpoint saver.
 * This is a thin wrapper around the synchronous implementation that
 * executes operations asynchronously.
 */
public class AsyncMemoryCheckpointSaver implements AsyncBaseCheckpointSaver {
    private final BaseCheckpointSaver synchronousSaver;
    
    /**
     * Create an async memory checkpoint saver.
     */
    public AsyncMemoryCheckpointSaver() {
        this.synchronousSaver = new MemoryCheckpointSaver();
    }
    
    /**
     * Create an async memory checkpoint saver with an existing synchronous saver.
     *
     * @param synchronousSaver The synchronous checkpoint saver to wrap
     */
    public AsyncMemoryCheckpointSaver(BaseCheckpointSaver synchronousSaver) {
        this.synchronousSaver = synchronousSaver;
    }
    
    @Override
    public CompletableFuture<String> checkpointAsync(String threadId, Map<String, Object> channelValues) {
        return CompletableFuture.supplyAsync(() -> 
            synchronousSaver.checkpoint(threadId, channelValues));
    }
    
    @Override
    public CompletableFuture<Optional<Map<String, Object>>> getValuesAsync(String checkpointId) {
        return CompletableFuture.supplyAsync(() -> 
            synchronousSaver.getValues(checkpointId));
    }
    
    @Override
    public CompletableFuture<List<String>> listAsync(String threadId) {
        return CompletableFuture.supplyAsync(() -> 
            synchronousSaver.list(threadId));
    }
    
    @Override
    public CompletableFuture<Optional<String>> latestAsync(String threadId) {
        return CompletableFuture.supplyAsync(() -> 
            synchronousSaver.latest(threadId));
    }
    
    @Override
    public CompletableFuture<Void> deleteAsync(String checkpointId) {
        return CompletableFuture.runAsync(() -> 
            synchronousSaver.delete(checkpointId));
    }
    
    @Override
    public CompletableFuture<Void> clearAsync(String threadId) {
        return CompletableFuture.runAsync(() -> 
            synchronousSaver.clear(threadId));
    }
    
    /**
     * Get the underlying synchronous checkpoint saver.
     *
     * @return The synchronous checkpoint saver
     */
    public BaseCheckpointSaver getSynchronousSaver() {
        return synchronousSaver;
    }
}