package com.langgraph.pregel.task;

import com.langgraph.pregel.retry.RetryPolicy;

import java.util.Objects;

/**
 * Represents a task to be executed within the Pregel system.
 * A task identifies a node to execute, an optional trigger, and a retry policy.
 */
public class PregelTask {
    private final String node;
    private final String trigger;
    private final RetryPolicy retryPolicy;
    
    /**
     * Create a PregelTask with all parameters.
     *
     * @param node Node name to execute
     * @param trigger Optional trigger that caused this task
     * @param retryPolicy Optional retry policy for execution failures
     */
    public PregelTask(String node, String trigger, RetryPolicy retryPolicy) {
        if (node == null || node.isEmpty()) {
            throw new IllegalArgumentException("Node name cannot be null or empty");
        }
        this.node = node;
        this.trigger = trigger;
        this.retryPolicy = retryPolicy;
    }
    
    /**
     * Create a PregelTask with just a node name.
     *
     * @param node Node name to execute
     */
    public PregelTask(String node) {
        this(node, null, null);
    }
    
    /**
     * Create a PregelTask with node name and trigger.
     *
     * @param node Node name to execute
     * @param trigger Trigger that caused this task
     */
    public PregelTask(String node, String trigger) {
        this(node, trigger, null);
    }
    
    /**
     * Get the node name.
     *
     * @return Node name
     */
    public String getNode() {
        return node;
    }
    
    /**
     * Get the trigger.
     *
     * @return Trigger or null if not triggered
     */
    public String getTrigger() {
        return trigger;
    }
    
    /**
     * Get the retry policy.
     *
     * @return RetryPolicy or null if using default policy
     */
    public RetryPolicy getRetryPolicy() {
        return retryPolicy;
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PregelTask that = (PregelTask) o;
        return Objects.equals(node, that.node) && 
               Objects.equals(trigger, that.trigger);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(node, trigger);
    }
    
    @Override
    public String toString() {
        return "PregelTask{" +
                "node='" + node + '\'' +
                (trigger != null ? ", trigger='" + trigger + '\'' : "") +
                '}';
    }
}