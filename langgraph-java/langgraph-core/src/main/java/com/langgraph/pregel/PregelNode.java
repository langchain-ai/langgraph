package com.langgraph.pregel;

import com.langgraph.pregel.channel.ChannelWriteEntry;
import com.langgraph.pregel.retry.RetryPolicy;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents an actor (node) in the Pregel system.
 * A node is a computational unit that subscribes to channels for inputs,
 * executes an action, and writes results to output channels.
 */
public class PregelNode {
    private final String name;
    private final PregelExecutable action;
    private final Set<String> subscribe;
    private final String trigger;
    private final List<ChannelWriteEntry> writers;
    private final RetryPolicy retryPolicy;
    
    /**
     * Create a PregelNode with write entries for outputs.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param subscribe Channel names this node listens to for updates
     * @param trigger Special condition for node execution
     * @param writeEntries Channel write entries that specify how to write outputs
     * @param retryPolicy Strategy for handling execution failures
     */
    public PregelNode(
            String name,
            PregelExecutable action,
            Collection<String> subscribe,
            String trigger,
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
        this.subscribe = subscribe != null ? new HashSet<>(subscribe) : Collections.emptySet();
        this.trigger = trigger;
        this.writers = writeEntries != null ? new ArrayList<>(writeEntries) : Collections.emptyList();
        this.retryPolicy = retryPolicy;
    }
    
    /**
     * Create a PregelNode with simple string channel names for outputs.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param subscribe Channel names this node listens to for updates
     * @param trigger Special condition for node execution
     * @param outputChannels Channel names to write outputs
     * @param retryPolicy Strategy for handling execution failures
     */
    public static PregelNode fromOutputChannels(
            String name,
            PregelExecutable action,
            Collection<String> subscribe,
            String trigger,
            Collection<String> outputChannels,
            RetryPolicy retryPolicy) {
        
        List<ChannelWriteEntry> writeEntries = outputChannels != null ? 
                outputChannels.stream()
                       .map(ChannelWriteEntry::new)
                       .collect(Collectors.toList()) : 
                Collections.emptyList();
                
        return new PregelNode(name, action, subscribe, trigger, writeEntries, retryPolicy);
    }
    
    /**
     * Create a PregelNode with just name and action.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     */
    public PregelNode(String name, PregelExecutable action) {
        this(name, action, null, null, (Collection<ChannelWriteEntry>) null, null);
    }
    
    /**
     * Create a PregelNode with name, action, and subscriptions.
     *
     * @param name Unique identifier for the node
     * @param action Function to execute when the node is triggered
     * @param subscribe Channel names this node listens to for updates
     */
    public PregelNode(String name, PregelExecutable action, Collection<String> subscribe) {
        this(name, action, subscribe, null, (Collection<ChannelWriteEntry>) null, null);
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
     * Get the channels this node subscribes to.
     *
     * @return Set of channel names (immutable)
     */
    public Set<String> getSubscribe() {
        return Collections.unmodifiableSet(subscribe);
    }
    
    /**
     * Get the trigger condition for this node.
     *
     * @return Trigger condition or null if not triggered
     */
    public String getTrigger() {
        return trigger;
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
     * Check if this node subscribes to a specific channel.
     *
     * @param channelName Channel name to check
     * @return True if the node subscribes to the channel
     */
    public boolean subscribesTo(String channelName) {
        return subscribe.contains(channelName);
    }
    
    /**
     * Check if this node has a specific trigger.
     *
     * @param triggerName Trigger name to check
     * @return True if the node has the trigger
     */
    public boolean hasTrigger(String triggerName) {
        return trigger != null && trigger.equals(triggerName);
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
                ", subscribes=" + subscribe +
                (trigger != null ? ", trigger='" + trigger + '\'' : "") +
                ", writers=" + writers +
                '}';
    }
    
    /**
     * Builder for creating PregelNode instances.
     */
    public static class Builder {
        private final String name;
        private final PregelExecutable action;
        private Set<String> subscribe = new HashSet<>();
        private String trigger;
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
         * Add a subscription to a channel.
         *
         * @param channelName Channel name to subscribe to
         * @return This builder
         */
        public Builder subscribe(String channelName) {
            if (channelName != null && !channelName.isEmpty()) {
                subscribe.add(channelName);
            }
            return this;
        }
        
        /**
         * Add multiple subscriptions.
         *
         * @param channelNames Channel names to subscribe to
         * @return This builder
         */
        public Builder subscribeAll(Collection<String> channelNames) {
            if (channelNames != null) {
                channelNames.forEach(this::subscribe);
            }
            return this;
        }
        
        /**
         * Set the trigger.
         *
         * @param trigger Trigger condition
         * @return This builder
         */
        public Builder trigger(String trigger) {
            this.trigger = trigger;
            return this;
        }
        
        /**
         * Add a writer entry.
         *
         * @param writeEntry Channel write entry
         * @return This builder
         */
        public Builder writer(ChannelWriteEntry writeEntry) {
            if (writeEntry != null) {
                writers.add(writeEntry);
            }
            return this;
        }
        
        /**
         * Add a simple writer for backward compatibility.
         *
         * @param channelName Channel name this node can write to
         * @return This builder
         */
        public Builder writer(String channelName) {
            if (channelName != null && !channelName.isEmpty()) {
                writers.add(new ChannelWriteEntry(channelName));
            }
            return this;
        }
        
        /**
         * Add multiple writer entries.
         *
         * @param writeEntries Collection of channel write entries
         * @return This builder
         */
        public Builder writeAll(Collection<ChannelWriteEntry> writeEntries) {
            if (writeEntries != null) {
                writeEntries.forEach(this::writer);
            }
            return this;
        }
        
        /**
         * Add multiple simple writers for backward compatibility.
         *
         * @param writerNames Channel names this node can write to
         * @return This builder
         */
        public Builder writeAllNames(Collection<String> writerNames) {
            if (writerNames != null) {
                writerNames.forEach(this::writer);
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
            return new PregelNode(name, action, subscribe, trigger, writers, retryPolicy);
        }
    }
}