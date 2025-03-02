package com.langgraph.pregel;

import com.langgraph.pregel.channel.ChannelWriteEntry;
import com.langgraph.pregel.retry.RetryPolicy;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents an actor (node) in the Pregel system.
 * A node is a computational unit that reads from input channels,
 * executes an action, and writes results to output channels.
 * 
 * <p>There are two key concepts for how nodes interact with channels:
 * <ul>
 *   <li>Input Channels ({@link #channels}): Channels from which the node reads values.
 *       When a node executes, it receives values from all its input channels.
 *   </li>
 *   <li>Trigger Channels ({@link #triggerChannels}): Special channel(s) that determine when this node
 *       should execute. A node will execute when any of its trigger channels are updated.
 *   </li>
 * </ul>
 * </p>
 * 
 * <p>In Python LangGraph, nodes only run on the first superstep if they have the input channel
 * as one of their triggers. In Java LangGraph, we now match this behavior - nodes only run
 * in the first superstep if they have appropriate trigger channels defined. For proper 
 * Python compatibility, it's important to explicitly define input channel as a trigger on 
 * nodes that should execute first.
 * </p>
 */
public class PregelNode {
    private final String name;
    private final PregelExecutable action;
    private final Set<String> channels;         // Input channels (formerly "subscribe")
    private final Set<String> triggerChannels;  // Trigger channels (formerly "trigger")
    private final List<ChannelWriteEntry> writers;
    private final RetryPolicy retryPolicy;
    
    /**
     * Create a PregelNode with write entries for outputs.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param channels Channel names this node reads values from
     * @param triggerChannels Channel(s) that determine when this node executes
     * @param writeEntries Channel write entries that specify how to write outputs
     * @param retryPolicy Strategy for handling execution failures
     */
    public PregelNode(
            String name,
            PregelExecutable action,
            Collection<String> channels,
            Collection<String> triggerChannels,
            Collection<ChannelWriteEntry> writeEntries,
            RetryPolicy retryPolicy) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Node name cannot be null or empty");
        }
        if (action == null) {
            throw new IllegalArgumentException("Action cannot be null");
        }
        
        this.name = name;
        this.action = action;
        this.channels = channels != null ? new HashSet<>(channels) : Collections.emptySet();
        this.triggerChannels = triggerChannels != null ? new HashSet<>(triggerChannels) : Collections.emptySet();
        this.writers = writeEntries != null ? new ArrayList<>(writeEntries) : Collections.emptyList();
        this.retryPolicy = retryPolicy;
    }
    
    /**
     * Create a PregelNode with just name and action.
     * For more complex configurations, use the Builder pattern.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     */
    public PregelNode(String name, PregelExecutable action) {
        this(name, action, null, null, (Collection<ChannelWriteEntry>) null, null);
    }
    
    /**
     * Create a PregelNode with name, action, and input channels.
     * This constructor exists primarily for testing purposes.
     * For more complex configurations, use the Builder pattern.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param channels Channel names this node reads values from
     */
    public PregelNode(String name, PregelExecutable action, Collection<String> channels) {
        this(name, action, channels, null, (Collection<ChannelWriteEntry>) null, null);
    }
    
    /**
     * Get the name of the node.
     *
     * @return Node name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the action to execute.
     *
     * @return Node action
     */
    public PregelExecutable getAction() {
        return action;
    }
    
    /**
     * Get the input channels this node reads from.
     *
     * @return Set of channel names (immutable)
     */
    public Set<String> getChannels() {
        return Collections.unmodifiableSet(channels);
    }
    
    /**
     * Get the trigger channels for this node.
     *
     * @return Set of trigger channels (immutable)
     */
    public Set<String> getTriggerChannels() {
        return Collections.unmodifiableSet(triggerChannels);
    }
    
    
    /**
     * Get the write entries for this node.
     *
     * @return List of channel write entries (immutable)
     */
    public List<ChannelWriteEntry> getWriteEntries() {
        return Collections.unmodifiableList(writers);
    }
    
    /**
     * Get the channels this node can write to.
     *
     * @return Set of channel names (immutable)
     */
    public Set<String> getWriters() {
        return writers.stream()
                .map(ChannelWriteEntry::getChannel)
                .collect(Collectors.toSet());
    }
    
    /**
     * Get the retry policy for this node.
     *
     * @return Retry policy or null if using default policy
     */
    public RetryPolicy getRetryPolicy() {
        return retryPolicy;
    }
    
    /**
     * Check if this node reads from a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node reads from the channel
     */
    public boolean readsFrom(String channelName) {
        return channels.contains(channelName);
    }
    
    /**
     * Check if this node is triggered by a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node is triggered by the channel
     */
    public boolean isTriggeredBy(String channelName) {
        return triggerChannels.contains(channelName);
    }
    
    
    /**
     * Check if this node can write to a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node can write to the channel
     */
    public boolean canWriteTo(String channelName) {
        return writers.stream()
                .anyMatch(entry -> entry.getChannel().equals(channelName));
    }
    
    /**
     * Find a write entry for a specific channel.
     *
     * @param channelName Channel name to look for
     * @return Optional write entry for the channel
     */
    public Optional<ChannelWriteEntry> getWriteEntry(String channelName) {
        return writers.stream()
                .filter(entry -> entry.getChannel().equals(channelName))
                .findFirst();
    }
    
    /**
     * Process node output according to write entries.
     *
     * @param nodeOutput Output from node execution
     * @return Processed output with values transformed as specified by write entries
     */
    public Map<String, Object> processOutput(Map<String, Object> nodeOutput) {
        if (nodeOutput == null || nodeOutput.isEmpty()) {
            return Collections.emptyMap();
        }
        
        Map<String, Object> result = new HashMap<>();
        
        // Process specific channel outputs
        for (ChannelWriteEntry entry : writers) {
            String channelName = entry.getChannel();
            Object value = entry.isPassthrough() ? nodeOutput.get(channelName) : entry.getValue();
            
            // Skip if explicit value is not found and this is a passthrough entry
            if (entry.isPassthrough() && !nodeOutput.containsKey(channelName)) {
                continue;
            }
            
            // Apply mapper if present
            if (entry.hasMapper()) {
                value = entry.getMapper().apply(value);
            }
            
            // Skip null values if configured to do so
            if (value == null && entry.isSkipNone()) {
                continue;
            }
            
            result.put(channelName, value);
        }
        
        // If no write entries are specified, pass through all outputs
        if (writers.isEmpty()) {
            result.putAll(nodeOutput);
        }
        
        return result;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PregelNode that = (PregelNode) o;
        return Objects.equals(name, that.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name);
    }
    
    @Override
    public String toString() {
        return "PregelNode{" +
                "name='" + name + '\'' +
                ", channels=" + channels +
                ", triggerChannels=" + triggerChannels +
                ", writers=" + writers +
                '}';
    }
    
    /**
     * Builder for creating PregelNode instances.
     */
    public static class Builder {
        private final String name;
        private final PregelExecutable action;
        private Set<String> channels = new HashSet<>();
        private Set<String> triggerChannels = new HashSet<>();
        private List<ChannelWriteEntry> writers = new ArrayList<>();
        private RetryPolicy retryPolicy;
        
        /**
         * Create a Builder with the required name and action.
         *
         * @param name Unique identifier for the node
         * @param action Function to execute when the node is triggered
         */
        public Builder(String name, PregelExecutable action) {
            if (name == null || name.isEmpty()) {
                throw new IllegalArgumentException("Node name cannot be null or empty");
            }
            if (action == null) {
                throw new IllegalArgumentException("Action cannot be null");
            }
            this.name = name;
            this.action = action;
        }
        
        /**
         * Add input channels that this node will read from.
         *
         * @param channelNames Channel names to read from (can be a single name or multiple names)
         * @return This builder
         */
        public Builder channels(Collection<String> channelNames) {
            if (channelNames != null) {
                for (String channelName : channelNames) {
                    if (channelName != null && !channelName.isEmpty()) {
                        channels.add(channelName);
                    }
                }
            }
            return this;
        }
        
        /**
         * Add a single input channel that this node will read from.
         *
         * @param channelName Channel name to read from
         * @return This builder
         */
        public Builder channels(String channelName) {
            if (channelName != null && !channelName.isEmpty()) {
                channels.add(channelName);
            }
            return this;
        }
        
        /**
         * Add trigger channels that determine when this node executes.
         *
         * @param channelNames Channel names that trigger execution (can be a single name or multiple names)
         * @return This builder
         */
        public Builder triggerChannels(Collection<String> channelNames) {
            if (channelNames != null) {
                for (String channelName : channelNames) {
                    if (channelName != null && !channelName.isEmpty()) {
                        triggerChannels.add(channelName);
                    }
                }
            }
            return this;
        }
        
        /**
         * Add a single trigger channel that determines when this node executes.
         *
         * @param channelName Channel name that triggers execution
         * @return This builder
         */
        public Builder triggerChannels(String channelName) {
            if (channelName != null && !channelName.isEmpty()) {
                triggerChannels.add(channelName);
            }
            return this;
        }
        
        
        /**
         * Add writers that specify where this node will write its output.
         *
         * @param entries Collection of ChannelWriteEntry objects
         * @return This builder
         */
        public Builder writers(Collection<ChannelWriteEntry> entries) {
            if (entries != null) {
                for (ChannelWriteEntry entry : entries) {
                    if (entry != null) {
                        writers.add(entry);
                    }
                }
            }
            return this;
        }
        
        /**
         * Add a single writer that specifies where this node will write its output.
         *
         * @param entry ChannelWriteEntry object
         * @return This builder
         */
        public Builder writers(ChannelWriteEntry entry) {
            if (entry != null) {
                writers.add(entry);
            }
            return this;
        }
        
        /**
         * Add a simple writer to the specified channel.
         * The node's output value for this channel will be passed through.
         *
         * @param channelName Channel name this node can write to
         * @return This builder
         */
        public Builder writers(String channelName) {
            if (channelName != null && !channelName.isEmpty()) {
                writers.add(new ChannelWriteEntry(channelName));
            }
            return this;
        }
        
        /**
         * Add multiple simple writers to the specified channels.
         * The node's output values for these channels will be passed through.
         *
         * @param channelNames Channel names this node can write to
         * @return This builder
         */
        public Builder writers(String... channelNames) {
            if (channelNames != null) {
                for (String name : channelNames) {
                    writers(name);
                }
            }
            return this;
        }
        
        /**
         * Add multiple simple writers from a collection of channel names.
         * The node's output values for these channels will be passed through.
         *
         * @param channelNames Collection of channel names this node can write to
         * @return This builder
         */
        public Builder writersFromCollection(Collection<String> channelNames) {
            if (channelNames != null) {
                for (String name : channelNames) {
                    writers(name);
                }
            }
            return this;
        }
        
        /**
         * Set the retry policy.
         *
         * @param retryPolicy Retry policy for handling failures
         * @return This builder
         */
        public Builder retryPolicy(RetryPolicy retryPolicy) {
            this.retryPolicy = retryPolicy;
            return this;
        }
        
        /**
         * Build the PregelNode.
         *
         * @return PregelNode instance
         */
        public PregelNode build() {
            return new PregelNode(name, action, channels, triggerChannels, writers, retryPolicy);
        }
    }
}