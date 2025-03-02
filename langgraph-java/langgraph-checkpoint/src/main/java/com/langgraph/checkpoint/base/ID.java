package com.langgraph.checkpoint.base;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.UUID;

/**
 * Utility class for generating deterministic IDs.
 */
public final class ID {
    private ID() {
        // Prevent instantiation
    }
    
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