package com.langgraph.pregel.channel;

import java.util.Objects;
import java.util.function.Predicate;

/**
 * Represents a permission to write to a channel with optional validation.
 * This defines both which channel a node can write to and rules for validating the writes.
 */
public class ChannelWritePermission {
    private final String channelName;
    private final Predicate<Object> validator;
    
    /**
     * Create a ChannelWritePermission with a validator.
     *
     * @param channelName Name of the channel
     * @param validator Optional validator for checking values written to the channel
     */
    public ChannelWritePermission(String channelName, Predicate<Object> validator) {
        if (channelName == null || channelName.isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        this.channelName = channelName;
        this.validator = validator;
    }
    
    /**
     * Create a ChannelWritePermission without a validator.
     *
     * @param channelName Name of the channel
     */
    public ChannelWritePermission(String channelName) {
        this(channelName, null);
    }
    
    /**
     * Get the channel name.
     *
     * @return Channel name
     */
    public String getChannelName() {
        return channelName;
    }
    
    /**
     * Get the validator.
     *
     * @return Validator or null if no validation is required
     */
    public Predicate<Object> getValidator() {
        return validator;
    }
    
    /**
     * Check if this permission has a validator.
     *
     * @return True if a validator is present
     */
    public boolean hasValidator() {
        return validator != null;
    }
    
    /**
     * Validate a value.
     *
     * @param value Value to validate
     * @return True if the value is valid or no validator is present
     */
    public boolean validate(Object value) {
        return validator == null || validator.test(value);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ChannelWritePermission that = (ChannelWritePermission) o;
        return Objects.equals(channelName, that.channelName);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(channelName);
    }
    
    @Override
    public String toString() {
        return "ChannelWritePermission{" +
                "channelName='" + channelName + '\'' +
                ", hasValidator=" + (validator != null) +
                '}';
    }
    
    /**
     * Create a builder for ChannelWritePermission.
     *
     * @param channelName Channel name
     * @return Builder
     */
    public static Builder builder(String channelName) {
        return new Builder(channelName);
    }
    
    /**
     * Builder for ChannelWritePermission.
     */
    public static class Builder {
        private final String channelName;
        private Predicate<Object> validator;
        
        /**
         * Create a Builder.
         *
         * @param channelName Channel name
         */
        public Builder(String channelName) {
            if (channelName == null || channelName.isEmpty()) {
                throw new IllegalArgumentException("Channel name cannot be null or empty");
            }
            this.channelName = channelName;
        }
        
        /**
         * Set the validator.
         *
         * @param validator Validator
         * @return This builder
         */
        public Builder validator(Predicate<Object> validator) {
            this.validator = validator;
            return this;
        }
        
        /**
         * Build the ChannelWritePermission.
         *
         * @return ChannelWritePermission
         */
        public ChannelWritePermission build() {
            return new ChannelWritePermission(channelName, validator);
        }
    }
}