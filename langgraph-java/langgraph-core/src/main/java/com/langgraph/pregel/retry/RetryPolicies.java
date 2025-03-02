package com.langgraph.pregel.retry;

import java.time.Duration;
import java.util.function.Predicate;

/**
 * Factory class for creating common retry policies.
 */
public final class RetryPolicies {
    private RetryPolicies() {
        // Prevent instantiation
    }
    
    /**
     * Create a retry policy that never retries.
     *
     * @return Retry policy
     */
    public static RetryPolicy noRetry() {
        return RetryPolicy.noRetry();
    }
    
    /**
     * Create a simple retry policy with a maximum number of attempts.
     *
     * @param maxAttempts Maximum number of attempts
     * @return Retry policy
     */
    public static RetryPolicy maxAttempts(int maxAttempts) {
        return RetryPolicy.maxAttempts(maxAttempts);
    }
    
    /**
     * Create a retry policy that always retries with a constant backoff.
     *
     * @param backoff Backoff duration between retries
     * @return Retry policy
     */
    public static RetryPolicy constantBackoff(Duration backoff) {
        return RetryPolicy.constantBackoff(backoff);
    }
    
    /**
     * Create a retry policy with exponential backoff.
     *
     * @param initialBackoff Initial backoff duration
     * @param maxAttempts Maximum number of attempts
     * @param maxBackoff Maximum backoff duration
     * @return Retry policy
     */
    public static RetryPolicy exponentialBackoff(Duration initialBackoff, int maxAttempts, Duration maxBackoff) {
        return RetryPolicy.exponentialBackoff(initialBackoff, maxAttempts, maxBackoff);
    }
    
    /**
     * Create a retry policy with exponential backoff and jitter.
     *
     * @param initialBackoff Initial backoff duration
     * @param maxAttempts Maximum number of attempts
     * @param maxBackoff Maximum backoff duration
     * @param jitterFactor Jitter factor (0.0 to 1.0, where 0.0 means no jitter)
     * @return Retry policy
     */
    public static RetryPolicy exponentialBackoffWithJitter(Duration initialBackoff, int maxAttempts,
                                                         Duration maxBackoff, double jitterFactor) {
        return RetryPolicy.exponentialBackoffWithJitter(initialBackoff, maxAttempts, maxBackoff, jitterFactor);
    }
    
    /**
     * Create a retry policy that filters exceptions.
     *
     * @param basePolicy Base retry policy to delegate to
     * @param filter Predicate to determine which exceptions should be retried
     * @return Retry policy
     */
    public static RetryPolicy withExceptionFilter(RetryPolicy basePolicy, Predicate<Throwable> filter) {
        return RetryPolicy.withExceptionFilter(basePolicy, filter);
    }
    
    /**
     * Create a retry policy that handles specific exception types.
     *
     * @param basePolicy Base retry policy to delegate to
     * @param exceptionClass Exception class to retry
     * @return Retry policy
     */
    public static <T extends Throwable> RetryPolicy onException(RetryPolicy basePolicy, Class<T> exceptionClass) {
        return withExceptionFilter(basePolicy, exceptionClass::isInstance);
    }
}