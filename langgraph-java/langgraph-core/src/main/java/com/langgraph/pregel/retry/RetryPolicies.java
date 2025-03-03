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
    
    /**
     * Create a builder for RetryPolicy.
     *
     * @return Builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for RetryPolicy.
     */
    public static class Builder {
        private int maxAttempts = 3; // Default value
        private Duration initialBackoff = Duration.ZERO;
        private Duration maxBackoff = Duration.ofSeconds(1);
        private double jitterFactor = 0.0;
        private Predicate<Throwable> exceptionFilter = throwable -> true;

        /**
         * Set the maximum number of attempts.
         *
         * @param maxAttempts Maximum number of attempts
         * @return This builder
         */
        public Builder maxAttempts(int maxAttempts) {
            this.maxAttempts = maxAttempts;
            return this;
        }

        /**
         * Set the initial backoff duration.
         *
         * @param initialBackoff Initial backoff duration
         * @return This builder
         */
        public Builder initialBackoff(Duration initialBackoff) {
            this.initialBackoff = initialBackoff;
            return this;
        }

        /**
         * Set the maximum backoff duration.
         *
         * @param maxBackoff Maximum backoff duration
         * @return This builder
         */
        public Builder maxBackoff(Duration maxBackoff) {
            this.maxBackoff = maxBackoff;
            return this;
        }

        /**
         * Set the jitter factor.
         *
         * @param jitterFactor Jitter factor (0.0 to 1.0, where 0.0 means no jitter)
         * @return This builder
         */
        public Builder jitterFactor(double jitterFactor) {
            this.jitterFactor = jitterFactor;
            return this;
        }

        /**
         * Set the exception filter.
         *
         * @param exceptionFilter Predicate to determine which exceptions should be retried
         * @return This builder
         */
        public Builder exceptionFilter(Predicate<Throwable> exceptionFilter) {
            this.exceptionFilter = exceptionFilter;
            return this;
        }

        /**
         * Build the RetryPolicy.
         *
         * @return RetryPolicy
         */
        public RetryPolicy build() {
            RetryPolicy basePolicy;
            
            if (initialBackoff.equals(Duration.ZERO)) {
                basePolicy = RetryPolicy.maxAttempts(maxAttempts);
            } else if (jitterFactor > 0) {
                basePolicy = RetryPolicy.exponentialBackoffWithJitter(
                    initialBackoff, maxAttempts, maxBackoff, jitterFactor);
            } else {
                basePolicy = RetryPolicy.exponentialBackoff(
                    initialBackoff, maxAttempts, maxBackoff);
            }
            
            if (exceptionFilter != null) {
                // Create a predicate that always returns true
                Predicate<Throwable> alwaysTrue = t -> true;
                
                // If the filter is different from the always-true predicate, apply it
                if (!exceptionFilter.equals(alwaysTrue)) {
                    return RetryPolicy.withExceptionFilter(basePolicy, exceptionFilter);
                }
            }
            
            return basePolicy;
        }
    }
}