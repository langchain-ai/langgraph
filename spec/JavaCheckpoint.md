# Java Checkpoint Interfaces

This document defines the Java interfaces for the checkpoint layer of LangGraph, aligned with the Python implementation.

## `BaseCheckpointSaver` Interface

The `BaseCheckpointSaver` interface provides methods for creating, loading, and managing checkpoints.

```java
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
```

## `ID` Utility

A utility class for generating deterministic IDs, similar to the Python implementation.

```java
package com.langgraph.checkpoint.base;

import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;

/**
 * Utility class for generating IDs.
 */
public final class ID {
    private ID() {} // Prevent instantiation

    /**
     * Generate a deterministic UUID based on a namespace and name.
     *
     * @param namespace The namespace for the ID
     * @param name The name within the namespace
     * @return A UUID
     */
    public static UUID uuid(String namespace, String name) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-1");
            md.update(namespace.getBytes(StandardCharsets.UTF_8));
            md.update(name.getBytes(StandardCharsets.UTF_8));
            byte[] digest = md.digest();

            // Set the version (4) and variant (RFC4122) bits
            digest[6] = (byte) ((digest[6] & 0x0F) | 0x40);
            digest[8] = (byte) ((digest[8] & 0x3F) | 0x80);

            long msb = 0;
            long lsb = 0;

            for (int i = 0; i < 8; i++) {
                msb = (msb << 8) | (digest[i] & 0xff);
            }

            for (int i = 8; i < 16; i++) {
                lsb = (lsb << 8) | (digest[i] & 0xff);
            }

            return new UUID(msb, lsb);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-1 algorithm not available", e);
        }
    }

    /**
     * Generate a checkpoint ID.
     *
     * @param threadId The thread ID
     * @return A checkpoint ID
     */
    public static String checkpointId(String threadId) {
        return uuid("checkpoint", threadId + "/" + System.currentTimeMillis()).toString();
    }

    /**
     * Generate a URL-safe base64 encoded ID.
     *
     * @param namespace The namespace for the ID
     * @param name The name within the namespace
     * @return A URL-safe base64-encoded ID
     */
    public static String urlSafeId(String namespace, String name) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(namespace.getBytes(StandardCharsets.UTF_8));
            md.update(name.getBytes(StandardCharsets.UTF_8));
            byte[] digest = md.digest();

            return Base64.getUrlEncoder().withoutPadding().encodeToString(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 algorithm not available", e);
        }
    }
}
```
