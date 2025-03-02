package com.langgraph.pregel.channel;

import java.util.Objects;
import java.util.function.Function;

/**
 * Represents a specification for writing to a channel.
 * This defines both the channel to write to and how values should be processed before writing.
 */
public class ChannelWriteEntry {
    /**
     * Special marker value indicating that the node's output value should be passed through.
     */
    public static final Object PASSTHROUGH = new Object() {
        @Override
        public String toString() {
            return "PASSTHROUGH";
        }
    };
    
    private final String channel;
    private final Object value;
    private final boolean skipNone;
    private final Function<Object, Object> mapper;
    
    /**
     * Create a ChannelWriteEntry with all parameters.
     *
     * @param channel Channel name to write to
     * @param value Value to write, or PASSTHROUGH to use the input
     * @param skipNone Whether to skip writing if the value is null
     * @param mapper Function to transform the value before writing
     */
    public ChannelWriteEntry(
            String channel,
            Object value,
            boolean skipNone,
            Function<Object, Object> mapper) {
        if (channel == null || channel.isEmpty()) {
            throw new IllegalArgumentException("Channel name cannot be null or empty");
        }
        this.channel = channel;
        this.value = value;
        this.skipNone = skipNone;
        this.mapper = mapper;
    }
    
    /**
     * Create a ChannelWriteEntry with default parameters.
     *
     * @param channel Channel name to write to
     */
    public ChannelWriteEntry(String channel) {
        this(channel, PASSTHROUGH, false, null);
    }
    
    /**
     * Create a ChannelWriteEntry with a specific value.
     *
     * @param channel Channel name to write to
     * @param value Value to write
     */
    public ChannelWriteEntry(String channel, Object value) {
        this(channel, value, false, null);
    }
    
    /**
     * Get the channel name.
     *
     * @return Channel name
     */
    public String getChannel() {
        return channel;
    }
    
    /**
     * Get the value to write.
     *
     * @return Value or PASSTHROUGH
     */
    public Object getValue() {
        return value;
    }
    
    /**
     * Check if writing should be skipped for null values.
     *
     * @return True if null values should be skipped
     */
    public boolean isSkipNone() {
        return skipNone;
    }
    
    /**
     * Get the mapper function.
     *
     * @return Mapper or null if no mapping is required
     */
    public Function<Object, Object> getMapper() {
        return mapper;
    }
    
    /**
     * Check if this entry uses a passthrough value.
     *
     * @return True if the value is PASSTHROUGH
     */
    public boolean isPassthrough() {
        return PASSTHROUGH.equals(value);
    }
    
    /**
     * Check if this entry has a mapper.
     *
     * @return True if a mapper is present
     */
    public boolean hasMapper() {
        return mapper != null;
    }
    
    /**
     * Process a value according to this entry's configuration.
     *
     * @param inputValue Input value (used if this entry is passthrough)
     * @return Processed value to write, or null if writing should be skipped
     */
    public Object processValue(Object inputValue) {
        // Determine the base value (either fixed or passthrough)
        Object baseValue = isPassthrough() ? inputValue : value;
        
        // Apply mapper if present
        Object processedValue = hasMapper() ? mapper.apply(baseValue) : baseValue;
        
        // Skip null values if configured to do so
        if (skipNone && processedValue == null) {
            return null;
        }
        
        return processedValue;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ChannelWriteEntry that = (ChannelWriteEntry) o;
        return skipNone == that.skipNone &&
                Objects.equals(channel, that.channel) &&
                Objects.equals(value, that.value);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(channel, value, skipNone);
    }
    
    @Override
    public String toString() {
        return "ChannelWriteEntry{" +
                "channel='" + channel + '\'' +
                ", value=" + (isPassthrough() ? "PASSTHROUGH" : value) +
                ", skipNone=" + skipNone +
                ", hasMapper=" + (mapper != null) +
                '}';
    }
    
    /**
     * Create a builder for ChannelWriteEntry.
     *
     * @param channel Channel name
     * @return Builder
     */
    public static Builder builder(String channel) {
        return new Builder(channel);
    }
    
    /**
     * Builder for ChannelWriteEntry.
     */
    public static class Builder {
        private final String channel;
        private Object value = PASSTHROUGH;
        private boolean skipNone = false;
        private Function<Object, Object> mapper = null;
        
        /**
         * Create a Builder.
         *
         * @param channel Channel name
         */
        public Builder(String channel) {
            if (channel == null || channel.isEmpty()) {
                throw new IllegalArgumentException("Channel name cannot be null or empty");
            }
            this.channel = channel;
        }
        
        /**
         * Set the value.
         *
         * @param value Value
         * @return This builder
         */
        public Builder value(Object value) {
            this.value = value;
            return this;
        }
        
        /**
         * Set to passthrough mode.
         *
         * @return This builder
         */
        public Builder passthrough() {
            this.value = PASSTHROUGH;
            return this;
        }
        
        /**
         * Set whether to skip null values.
         *
         * @param skipNone Whether to skip null values
         * @return This builder
         */
        public Builder skipNone(boolean skipNone) {
            this.skipNone = skipNone;
            return this;
        }
        
        /**
         * Set the mapper function.
         *
         * @param mapper Mapper function
         * @return This builder
         */
        public Builder mapper(Function<Object, Object> mapper) {
            this.mapper = mapper;
            return this;
        }
        
        /**
         * Build the ChannelWriteEntry.
         *
         * @return ChannelWriteEntry
         */
        public ChannelWriteEntry build() {
            return new ChannelWriteEntry(channel, value, skipNone, mapper);
        }
    }
}