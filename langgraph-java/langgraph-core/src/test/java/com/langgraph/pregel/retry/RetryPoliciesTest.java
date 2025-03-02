package com.langgraph.pregel.retry;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

public class RetryPoliciesTest {
    
    private static final RuntimeException TEST_EXCEPTION = new RuntimeException("Test exception");
    
    @Test
    void testNoRetryPolicy() {
        RetryPolicy policy = RetryPolicies.noRetry();
        
        RetryPolicy.RetryDecision decision = policy.shouldRetry(1, TEST_EXCEPTION);
        
        assertThat(decision.shouldRetry()).isFalse();
        assertThat(decision.getBackoff()).isEqualTo(Duration.ZERO);
    }
    
    @Test
    void testMaxAttemptsPolicy() {
        RetryPolicy policy = RetryPolicies.maxAttempts(3);
        
        // First attempt (1) - should retry
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        
        // Third attempt (3) - should fail (max attempts reached)
        RetryPolicy.RetryDecision decision3 = policy.shouldRetry(3, TEST_EXCEPTION);
        assertThat(decision3.shouldRetry()).isFalse();
    }
    
    @Test
    void testConstantBackoffPolicy() {
        Duration backoff = Duration.ofMillis(100);
        RetryPolicy policy = RetryPolicies.constantBackoff(backoff);
        
        RetryPolicy.RetryDecision decision = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision.shouldRetry()).isTrue();
        assertThat(decision.getBackoff()).isEqualTo(backoff);
    }
    
    @Test
    void testExponentialBackoffPolicy() {
        Duration initialBackoff = Duration.ofMillis(100);
        Duration maxBackoff = Duration.ofSeconds(1);
        int maxAttempts = 5;
        
        RetryPolicy policy = RetryPolicies.exponentialBackoff(initialBackoff, maxAttempts, maxBackoff);
        
        // First attempt - should retry with initial backoff
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        assertThat(decision1.getBackoff()).isEqualTo(Duration.ofMillis(100));
        
        // Fourth attempt - backoff should be 800ms
        RetryPolicy.RetryDecision decision4 = policy.shouldRetry(4, TEST_EXCEPTION);
        assertThat(decision4.shouldRetry()).isTrue();
        assertThat(decision4.getBackoff()).isEqualTo(Duration.ofMillis(800));
    }
    
    @Test
    void testExponentialBackoffWithJitterPolicy() {
        Duration initialBackoff = Duration.ofMillis(100);
        Duration maxBackoff = Duration.ofSeconds(1);
        int maxAttempts = 3;
        double jitterFactor = 0.5;
        
        RetryPolicy policy = RetryPolicies.exponentialBackoffWithJitter(
                initialBackoff, maxAttempts, maxBackoff, jitterFactor);
        
        // First attempt
        RetryPolicy.RetryDecision decision = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision.shouldRetry()).isTrue();
        assertThat(decision.getBackoff().toMillis()).isBetween(
                (long)(initialBackoff.toMillis() * (1 - jitterFactor)),
                (long)(initialBackoff.toMillis() * (1 + jitterFactor))
        );
    }
    
    @Test
    void testWithExceptionFilterPolicy() {
        RetryPolicy basePolicy = RetryPolicies.maxAttempts(3);
        
        // Filter that only retries RuntimeException
        RetryPolicy policy = RetryPolicies.withExceptionFilter(
                basePolicy, e -> e instanceof RuntimeException);
        
        // Should retry RuntimeException
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, new RuntimeException());
        assertThat(decision1.shouldRetry()).isTrue();
        
        // Should not retry other exceptions
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(1, new Exception());
        assertThat(decision2.shouldRetry()).isFalse();
    }
    
    @Test
    void testOnExceptionPolicy() {
        RetryPolicy basePolicy = RetryPolicies.maxAttempts(3);
        
        // Only retry IOException
        RetryPolicy policy = RetryPolicies.onException(basePolicy, IOException.class);
        
        // Should retry IOException
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, new IOException());
        assertThat(decision1.shouldRetry()).isTrue();
        
        // Should not retry RuntimeException
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(1, new RuntimeException());
        assertThat(decision2.shouldRetry()).isFalse();
        
        // Should retry subclasses of specified exception
        RetryPolicy.RetryDecision decision3 = policy.shouldRetry(1, new java.io.FileNotFoundException());
        assertThat(decision3.shouldRetry()).isTrue();
    }
}