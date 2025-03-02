package com.langgraph.checkpoint.base;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Asynchronous interface for saving and loading checkpoints.
 */
public interface AsyncBaseCheckpointSaver {
    /**
     * Create a new checkpoint asynchronously.
     *
     * @param threadId The ID of the thread to checkpoint
     * @param channelValues The values of the channels to checkpoint
     * @return CompletableFuture with the ID of the new checkpoint
     */
    CompletableFuture<String> checkpointAsync(String threadId, Map<String, Object> channelValues);

    /**
     * Get values from a checkpoint asynchronously.
     *
     * @param checkpointId The ID of the checkpoint to load
     * @return CompletableFuture with the channel values from the checkpoint, or empty if not found
     */
    CompletableFuture<Optional<Map<String, Object>>> getValuesAsync(String checkpointId);

    /**
     * List all checkpoints for a thread asynchronously.
     *
     * @param threadId The ID of the thread
     * @return CompletableFuture with list of checkpoint IDs
     */
    CompletableFuture<List<String>> listAsync(String threadId);

    /**
     * Get the latest checkpoint for a thread asynchronously.
     *
     * @param threadId The ID of the thread
     * @return CompletableFuture with the ID of the latest checkpoint, or empty if none exists
     */
    CompletableFuture<Optional<String>> latestAsync(String threadId);
    
    /**
     * Delete a checkpoint asynchronously.
     *
     * @param checkpointId The ID of the checkpoint to delete
     * @return CompletableFuture completed when deletion is done
     */
    CompletableFuture<Void> deleteAsync(String checkpointId);
    
    /**
     * Clear all checkpoints for a thread asynchronously.
     *
     * @param threadId The ID of the thread
     * @return CompletableFuture completed when clearing is done
     */
    CompletableFuture<Void> clearAsync(String threadId);
}