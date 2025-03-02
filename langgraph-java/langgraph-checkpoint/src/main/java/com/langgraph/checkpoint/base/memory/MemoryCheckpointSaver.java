package com.langgraph.checkpoint.base.memory;

import com.langgraph.checkpoint.base.BaseCheckpointSaver;
import com.langgraph.checkpoint.base.ID;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory implementation of a checkpoint saver.
 */
public class MemoryCheckpointSaver implements BaseCheckpointSaver {
    private final Map<String, Map<String, Object>> checkpoints = new ConcurrentHashMap<>();
    private final Map<String, List<String>> threadCheckpoints = new ConcurrentHashMap<>();
    
    @Override
    public String checkpoint(String threadId, Map<String, Object> channelValues) {
        String checkpointId = ID.checkpointId(threadId);
        
        // Store the checkpoint
        checkpoints.put(checkpointId, new HashMap<>(channelValues));
        
        // Add to thread's checkpoints
        threadCheckpoints.computeIfAbsent(threadId, k -> 
            Collections.synchronizedList(new ArrayList<>())).add(checkpointId);
        
        return checkpointId;
    }
    
    @Override
    public Optional<Map<String, Object>> getValues(String checkpointId) {
        Map<String, Object> values = checkpoints.get(checkpointId);
        return Optional.ofNullable(values).map(HashMap::new);
    }
    
    @Override
    public List<String> list(String threadId) {
        List<String> result = threadCheckpoints.get(threadId);
        return result != null ? new ArrayList<>(result) : Collections.emptyList();
    }
    
    @Override
    public Optional<String> latest(String threadId) {
        List<String> checkpoints = threadCheckpoints.get(threadId);
        
        if (checkpoints == null || checkpoints.isEmpty()) {
            return Optional.empty();
        }
        
        return Optional.of(checkpoints.get(checkpoints.size() - 1));
    }
    
    @Override
    public void delete(String checkpointId) {
        // Remove the checkpoint
        Map<String, Object> removed = checkpoints.remove(checkpointId);
        
        if (removed != null) {
            // Find and remove from thread's checkpoints
            for (List<String> checkpointsList : threadCheckpoints.values()) {
                checkpointsList.remove(checkpointId);
            }
        }
    }
    
    @Override
    public void clear(String threadId) {
        List<String> checkpointIds = threadCheckpoints.remove(threadId);
        
        if (checkpointIds != null) {
            // Remove all checkpoints for this thread
            for (String checkpointId : checkpointIds) {
                checkpoints.remove(checkpointId);
            }
        }
    }
}