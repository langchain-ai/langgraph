package com.langgraph.checkpoint.base;

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Interface for saving and loading checkpoints.
 */
public interface BaseCheckpointSaver {
    /**
     * Create a new checkpoint.
     *
     * @param threadId The ID of the thread to checkpoint
     * @param channelValues The values of the channels to checkpoint
     * @return The ID of the new checkpoint
     */
    String checkpoint(String threadId, Map<String, Object> channelValues);

    /**
     * Get values from a checkpoint.
     *
     * @param checkpointId The ID of the checkpoint to load
     * @return The channel values from the checkpoint, or empty if not found
     */
    Optional<Map<String, Object>> getValues(String checkpointId);

    /**
     * List all checkpoints for a thread.
     *
     * @param threadId The ID of the thread
     * @return List of checkpoint IDs
     */
    List<String> list(String threadId);

    /**
     * Get the latest checkpoint for a thread.
     *
     * @param threadId The ID of the thread
     * @return The ID of the latest checkpoint, or empty if none exists
     */
    Optional<String> latest(String threadId);
    
    /**
     * Delete a checkpoint.
     *
     * @param checkpointId The ID of the checkpoint to delete
     */
    void delete(String checkpointId);
    
    /**
     * Clear all checkpoints for a thread.
     *
     * @param threadId The ID of the thread
     */
    void clear(String threadId);
}