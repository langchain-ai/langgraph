package com.langgraph.pregel.retry;

import java.time.Duration;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Predicate;

/**
 * Interface for handling execution failures by determining if and how to retry failed tasks.
 */
public interface RetryPolicy {
    /**
     * Decide how to handle a failed execution.
     *
     * @param attempt Current attempt number (1-based)
     * @param error Error that occurred
     * @return Retry decision with backoff information
     */
    RetryDecision shouldRetry(int attempt, Throwable error);
    
    /**
     * Class representing a retry decision.
     */
    class RetryDecision {
        private final boolean shouldRetry;
        private final Duration backoff;
        
        private RetryDecision(boolean shouldRetry, Duration backoff) {
            this.shouldRetry = shouldRetry;
            this.backoff = backoff;
        }
        
        /**
         * Check if the task should be retried.
         *
         * @return true if the task should be retried, false otherwise
         */
        public boolean shouldRetry() {
            return shouldRetry;
        }
        
        /**
         * Get the backoff duration before the next retry.
         *
         * @return Duration to wait before the next retry
         */
        public Duration getBackoff() {
            return backoff;
        }
        
        /**
         * Create a decision to retry after the specified backoff.
         *
         * @param backoff Duration to wait before the next retry
         * @return Retry decision
         */
        public static RetryDecision retry(Duration backoff) {
            return new RetryDecision(true, backoff);
        }
        
        /**
         * Create a decision to not retry.
         *
         * @return Retry decision
         */
        public static RetryDecision fail() {
            return new RetryDecision(false, Duration.ZERO);
        }
    }
    
    /**
     * Create a simple retry policy with a maximum number of attempts.
     *
     * @param maxAttempts Maximum number of attempts
     * @return Retry policy
     */
    static RetryPolicy maxAttempts(int maxAttempts) {
        return (attempt, error) -> 
            attempt < maxAttempts ? RetryDecision.retry(Duration.ZERO) : RetryDecision.fail();
    }
    
    /**
     * Create a retry policy that never retries.
     *
     * @return Retry policy
     */
    static RetryPolicy noRetry() {
        return (attempt, error) -> RetryDecision.fail();
    }
    
    /**
     * Create a retry policy that always retries with a constant backoff.
     *
     * @param backoff Backoff duration between retries
     * @return Retry policy
     */
    static RetryPolicy constantBackoff(Duration backoff) {
        return (attempt, error) -> RetryDecision.retry(backoff);
    }
    
    /**
     * Create a retry policy with exponential backoff.
     *
     * @param initialBackoff Initial backoff duration
     * @param maxAttempts Maximum number of attempts
     * @param maxBackoff Maximum backoff duration
     * @return Retry policy
     */
    static RetryPolicy exponentialBackoff(Duration initialBackoff, int maxAttempts, Duration maxBackoff) {
        return (attempt, error) -> {
            if (attempt >= maxAttempts) {
                return RetryDecision.fail();
            }
            
            long initialBackoffMillis = initialBackoff.toMillis();
            long maxBackoffMillis = maxBackoff.toMillis();
            
            // Calculate exponential backoff: initialBackoff * 2^(attempt-1)
            long backoffMillis = initialBackoffMillis * (1L << (attempt - 1));
            
            // Ensure backoff doesn't exceed maxBackoff
            backoffMillis = Math.min(backoffMillis, maxBackoffMillis);
            
            return RetryDecision.retry(Duration.ofMillis(backoffMillis));
        };
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
    static RetryPolicy exponentialBackoffWithJitter(Duration initialBackoff, int maxAttempts, 
                                                   Duration maxBackoff, double jitterFactor) {
        return (attempt, error) -> {
            if (attempt >= maxAttempts) {
                return RetryDecision.fail();
            }
            
            long initialBackoffMillis = initialBackoff.toMillis();
            long maxBackoffMillis = maxBackoff.toMillis();
            
            // Calculate exponential backoff: initialBackoff * 2^(attempt-1)
            long backoffMillis = initialBackoffMillis * (1L << (attempt - 1));
            
            // Ensure backoff doesn't exceed maxBackoff
            backoffMillis = Math.min(backoffMillis, maxBackoffMillis);
            
            if (jitterFactor > 0) {
                // Apply jitter: backoff * (1 - jitterFactor + random * 2 * jitterFactor)
                double jitter = 1.0 - jitterFactor + ThreadLocalRandom.current().nextDouble() * 2 * jitterFactor;
                backoffMillis = (long) (backoffMillis * jitter);
            }
            
            return RetryDecision.retry(Duration.ofMillis(backoffMillis));
        };
    }
    
    /**
     * Create a retry policy that filters exceptions.
     *
     * @param basePolicy Base retry policy to delegate to
     * @param filter Predicate to determine which exceptions should be retried
     * @return Retry policy
     */
    static RetryPolicy withExceptionFilter(RetryPolicy basePolicy, Predicate<Throwable> filter) {
        return (attempt, error) -> {
            if (filter.test(error)) {
                return basePolicy.shouldRetry(attempt, error);
            } else {
                return RetryDecision.fail();
            }
        };
    }
}