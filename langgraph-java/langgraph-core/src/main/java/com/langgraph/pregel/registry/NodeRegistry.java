package com.langgraph.pregel.registry;

import com.langgraph.pregel.PregelNode;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Registry for managing a collection of nodes.
 * Provides methods for registration, validation, and node lookup.
 */
public class NodeRegistry {
    private final Map<String, PregelNode> nodes;
    
    /**
     * Create an empty NodeRegistry.
     */
    public NodeRegistry() {
        this.nodes = new HashMap<>();
    }
    
    /**
     * Create a NodeRegistry with initial nodes.
     *
     * @param nodes Collection of nodes to register
     */
    public NodeRegistry(Collection<PregelNode> nodes) {
        this.nodes = new HashMap<>();
        if (nodes != null) {
            nodes.forEach(this::register);
        }
    }
    
    /**
     * Create a NodeRegistry with initial nodes.
     *
     * @param nodes Map of node names to nodes
     */
    public NodeRegistry(Map<String, PregelNode> nodes) {
        this.nodes = new HashMap<>();
        if (nodes != null) {
            nodes.forEach((name, node) -> {
                if (!name.equals(node.getName())) {
                    throw new IllegalArgumentException(
                            "Node name mismatch: key '" + name + "' != node name '" + node.getName() + "'");
                }
                register(node);
            });
        }
    }
    
    /**
     * Register a node.
     *
     * @param node Node to register
     * @return This registry
     * @throws IllegalArgumentException If a node with the same name is already registered
     */
    public NodeRegistry register(PregelNode node) {
        if (node == null) {
            throw new IllegalArgumentException("Node cannot be null");
        }
        
        String name = node.getName();
        if (nodes.containsKey(name)) {
            throw new IllegalArgumentException("Node with name '" + name + "' is already registered");
        }
        
        nodes.put(name, node);
        return this;
    }
    
    /**
     * Register multiple nodes.
     *
     * @param nodesToRegister Collection of nodes to register
     * @return This registry
     * @throws IllegalArgumentException If a node with the same name is already registered
     */
    public NodeRegistry registerAll(Collection<PregelNode> nodesToRegister) {
        if (nodesToRegister != null) {
            nodesToRegister.forEach(this::register);
        }
        return this;
    }
    
    /**
     * Get a node by name.
     *
     * @param name Name of the node to get
     * @return Node with the given name
     * @throws NoSuchElementException If no node with the given name is registered
     */
    public PregelNode get(String name) {
        PregelNode node = nodes.get(name);
        if (node == null) {
            throw new NoSuchElementException("No node registered with name '" + name + "'");
        }
        return node;
    }
    
    /**
     * Check if a node with the given name is registered.
     *
     * @param name Name to check
     * @return True if a node with the given name is registered
     */
    public boolean contains(String name) {
        return nodes.containsKey(name);
    }
    
    /**
     * Remove a node by name.
     *
     * @param name Name of the node to remove
     * @return This registry
     */
    public NodeRegistry remove(String name) {
        nodes.remove(name);
        return this;
    }
    
    /**
     * Get all registered nodes.
     *
     * @return Unmodifiable map of node names to nodes
     */
    public Map<String, PregelNode> getAll() {
        return Collections.unmodifiableMap(nodes);
    }
    
    /**
     * Get the number of registered nodes.
     *
     * @return Number of registered nodes
     */
    public int size() {
        return nodes.size();
    }
    
    /**
     * Get all nodes that subscribe to the given channel.
     *
     * @param channelName Channel name
     * @return Set of nodes that subscribe to the channel
     */
    public Set<PregelNode> getSubscribers(String channelName) {
        return nodes.values().stream()
                .filter(node -> node.subscribesTo(channelName))
                .collect(Collectors.toSet());
    }
    
    /**
     * Get all nodes that have the given trigger.
     *
     * @param triggerName Trigger name
     * @return Set of nodes that have the trigger
     */
    public Set<PregelNode> getTriggered(String triggerName) {
        return nodes.values().stream()
                .filter(node -> node.hasTrigger(triggerName))
                .collect(Collectors.toSet());
    }
    
    /**
     * Get all nodes that can write to the given channel.
     *
     * @param channelName Channel name
     * @return Set of nodes that can write to the channel
     */
    public Set<PregelNode> getWriters(String channelName) {
        return nodes.values().stream()
                .filter(node -> node.canWriteTo(channelName))
                .collect(Collectors.toSet());
    }
    
    /**
     * Validate the registry.
     * Checks that all nodes have valid configurations.
     *
     * @throws IllegalStateException If the registry is invalid
     */
    public void validate() {
        // Validate that each node has a unique name
        Set<String> nodeNames = new HashSet<>();
        for (PregelNode node : nodes.values()) {
            String name = node.getName();
            if (nodeNames.contains(name)) {
                throw new IllegalStateException("Duplicate node name: " + name);
            }
            nodeNames.add(name);
        }
    }
    
    /**
     * Validate that nodes only subscribe to existing channels.
     *
     * @param channelNames Set of valid channel names
     * @throws IllegalStateException If a node subscribes to a non-existent channel
     */
    public void validateSubscriptions(Set<String> channelNames) {
        for (PregelNode node : nodes.values()) {
            for (String channelName : node.getSubscribe()) {
                if (!channelNames.contains(channelName)) {
                    throw new IllegalStateException(
                            "Node '" + node.getName() + "' subscribes to non-existent channel '" + channelName + "'");
                }
            }
        }
    }
    
    /**
     * Validate that nodes only write to existing channels.
     *
     * @param channelNames Set of valid channel names
     * @throws IllegalStateException If a node writes to a non-existent channel
     */
    public void validateWriters(Set<String> channelNames) {
        for (PregelNode node : nodes.values()) {
            for (String channelName : node.getWriters()) {
                if (!channelNames.contains(channelName)) {
                    throw new IllegalStateException(
                            "Node '" + node.getName() + "' writes to non-existent channel '" + channelName + "'");
                }
            }
        }
    }
    
    /**
     * Validate that nodes only use existing triggers.
     *
     * @param channelNames Set of valid channel names
     * @throws IllegalStateException If a node uses a non-existent trigger
     */
    public void validateTriggers(Set<String> channelNames) {
        for (PregelNode node : nodes.values()) {
            String trigger = node.getTrigger();
            if (trigger != null && !channelNames.contains(trigger)) {
                throw new IllegalStateException(
                        "Node '" + node.getName() + "' has non-existent trigger '" + trigger + "'");
            }
        }
    }
}