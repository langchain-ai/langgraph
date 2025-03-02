package com.langgraph.pregel.task;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.HashMap;

/**
 * Represents an executable task with inputs and context.
 * This is a concrete task ready for execution with all required data.
 */
public class PregelExecutableTask {
    private final PregelTask task;
    private final Map<String, Object> inputs;
    private final Map<String, Object> context;
    
    /**
     * Create a PregelExecutableTask with all parameters.
     *
     * @param task Task to execute
     * @param inputs Channel inputs for the task
     * @param context Execution context
     */
    public PregelExecutableTask(
            PregelTask task,
            Map<String, Object> inputs,
            Map<String, Object> context) {
        if (task == null) {
            throw new IllegalArgumentException("Task cannot be null");
        }
        this.task = task;
        this.inputs = inputs != null ? new HashMap<>(inputs) : Collections.emptyMap();
        this.context = context != null ? new HashMap<>(context) : Collections.emptyMap();
    }
    
    /**
     * Get the task.
     *
     * @return Task
     */
    public PregelTask getTask() {
        return task;
    }
    
    /**
     * Get the inputs.
     *
     * @return Map of channel inputs (immutable)
     */
    public Map<String, Object> getInputs() {
        return Collections.unmodifiableMap(inputs);
    }
    
    /**
     * Get the context.
     *
     * @return Map of context values (immutable)
     */
    public Map<String, Object> getContext() {
        return Collections.unmodifiableMap(context);
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PregelExecutableTask that = (PregelExecutableTask) o;
        return Objects.equals(task, that.task) &&
               Objects.equals(inputs, that.inputs) &&
               Objects.equals(context, that.context);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(task, inputs, context);
    }
    
    @Override
    public String toString() {
        return "PregelExecutableTask{" +
                "task=" + task +
                ", inputs=" + inputs.keySet() +
                ", context=" + context.keySet() +
                '}';
    }
}