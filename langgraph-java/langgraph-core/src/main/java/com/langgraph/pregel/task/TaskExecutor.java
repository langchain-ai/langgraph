package com.langgraph.pregel.task;

import com.langgraph.pregel.PregelNode;
import com.langgraph.pregel.retry.RetryPolicy;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

/**
 * Executes PregelExecutableTask instances with retry logic.
 */
public class TaskExecutor {
    private static final int DEFAULT_MAX_ATTEMPTS = 3;
    private final RetryPolicy defaultRetryPolicy;
    
    /**
     * Create a TaskExecutor with a default retry policy.
     *
     * @param defaultRetryPolicy Default retry policy for tasks
     */
    public TaskExecutor(RetryPolicy defaultRetryPolicy) {
        this.defaultRetryPolicy = defaultRetryPolicy;
    }
    
    /**
     * Create a TaskExecutor with a default retry policy allowing up to 3 attempts.
     */
    public TaskExecutor() {
        this(RetryPolicy.maxAttempts(DEFAULT_MAX_ATTEMPTS));
    }
    
    /**
     * Execute a task.
     *
     * @param node Node to execute
     * @param task Task to execute
     * @return Result of the execution
     * @throws TaskExecutionException If execution fails after all retry attempts
     */
    public Map<String, Object> execute(PregelNode node, PregelExecutableTask task) throws TaskExecutionException {
        if (node == null) {
            throw new IllegalArgumentException("Node cannot be null");
        }
        if (task == null) {
            throw new IllegalArgumentException("Task cannot be null");
        }
        
        // Get the appropriate retry policy
        RetryPolicy retryPolicy = task.getTask().getRetryPolicy();
        if (retryPolicy == null) {
            retryPolicy = node.getRetryPolicy();
            if (retryPolicy == null) {
                retryPolicy = defaultRetryPolicy;
            }
        }
        
        // Execute with retry
        return executeWithRetry(() -> {
            Map<String, Object> inputs = task.getInputs();
            Map<String, Object> context = task.getContext();
            
            try {
                // Execute the action
                return node.getAction().execute(inputs, context);
            } catch (Exception e) {
                // Wrap and rethrow
                throw new TaskExecutionException("Error executing node " + node.getName(), e);
            }
        }, retryPolicy);
    }
    
    /**
     * Execute a task asynchronously.
     *
     * @param node Node to execute
     * @param task Task to execute
     * @return CompletableFuture with the result of the execution
     */
    public CompletableFuture<Map<String, Object>> executeAsync(PregelNode node, PregelExecutableTask task) {
        return CompletableFuture.supplyAsync(() -> execute(node, task));
    }
    
    /**
     * Execute a callable with retry logic.
     *
     * @param <T> Type of the result
     * @param callable Callable to execute
     * @param retryPolicy Retry policy to use
     * @return Result of the callable
     * @throws TaskExecutionException If execution fails after all retry attempts
     */
    private <T> T executeWithRetry(Callable<T> callable, RetryPolicy retryPolicy) throws TaskExecutionException {
        int attempt = 1;
        Throwable lastError = null;
        
        while (true) {
            try {
                return callable.call();
            } catch (CancellationException | InterruptedException e) {
                // Do not retry cancellation or interruption
                Thread.currentThread().interrupt();
                throw new TaskExecutionException("Task execution was cancelled or interrupted", e);
            } catch (CompletionException e) {
                // Unwrap CompletionException
                lastError = e.getCause() != null ? e.getCause() : e;
            } catch (Exception e) {
                lastError = e;
            }
            
            // If we get here, execution failed
            if (retryPolicy != null) {
                RetryPolicy.RetryDecision decision = retryPolicy.shouldRetry(attempt, lastError);
                
                if (decision.shouldRetry()) {
                    // Sleep if backoff is specified
                    Duration backoff = decision.getBackoff();
                    if (!backoff.isZero() && !backoff.isNegative()) {
                        try {
                            Thread.sleep(backoff.toMillis());
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            throw new TaskExecutionException("Retry was interrupted", ie);
                        }
                    }
                    
                    // Increment attempt counter
                    attempt++;
                } else {
                    // Do not retry
                    break;
                }
            } else {
                // No retry policy, fail immediately
                break;
            }
        }
        
        // All retries failed or no retry policy
        throw new TaskExecutionException("Task execution failed after " + attempt + " attempts", lastError);
    }
}