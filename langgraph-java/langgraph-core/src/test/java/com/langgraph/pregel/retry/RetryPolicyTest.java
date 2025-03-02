package com.langgraph.pregel.retry;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.time.Duration;
import java.util.concurrent.ThreadLocalRandom;

import static org.assertj.core.api.Assertions.assertThat;

public class RetryPolicyTest {
    
    private static final RuntimeException TEST_EXCEPTION = new RuntimeException("Test exception");
    
    @Test
    void testNoRetryPolicy() {
        RetryPolicy policy = RetryPolicy.noRetry();
        
        RetryPolicy.RetryDecision decision = policy.shouldRetry(1, TEST_EXCEPTION);
        
        assertThat(decision.shouldRetry()).isFalse();
        assertThat(decision.getBackoff()).isEqualTo(Duration.ZERO);
    }
    
    @Test
    void testMaxAttemptsPolicy() {
        RetryPolicy policy = RetryPolicy.maxAttempts(3);
        
        // First attempt (1) - should retry
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        assertThat(decision1.getBackoff()).isEqualTo(Duration.ZERO);
        
        // Second attempt (2) - should retry
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(2, TEST_EXCEPTION);
        assertThat(decision2.shouldRetry()).isTrue();
        assertThat(decision2.getBackoff()).isEqualTo(Duration.ZERO);
        
        // Third attempt (3) - should fail (max attempts reached)
        RetryPolicy.RetryDecision decision3 = policy.shouldRetry(3, TEST_EXCEPTION);
        assertThat(decision3.shouldRetry()).isFalse();
        assertThat(decision3.getBackoff()).isEqualTo(Duration.ZERO);
    }
    
    @Test
    void testConstantBackoffPolicy() {
        Duration backoff = Duration.ofMillis(100);
        RetryPolicy policy = RetryPolicy.constantBackoff(backoff);
        
        // First attempt - should retry with constant backoff
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        assertThat(decision1.getBackoff()).isEqualTo(backoff);
        
        // Second attempt - should retry with same constant backoff
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(2, TEST_EXCEPTION);
        assertThat(decision2.shouldRetry()).isTrue();
        assertThat(decision2.getBackoff()).isEqualTo(backoff);
    }
    
    @Test
    void testExponentialBackoffPolicy() {
        Duration initialBackoff = Duration.ofMillis(100);
        Duration maxBackoff = Duration.ofSeconds(1);
        int maxAttempts = 5;
        
        RetryPolicy policy = RetryPolicy.exponentialBackoff(initialBackoff, maxAttempts, maxBackoff);
        
        // First attempt - should retry with initial backoff
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        assertThat(decision1.getBackoff()).isEqualTo(Duration.ofMillis(100));
        
        // Second attempt - should retry with doubled backoff
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(2, TEST_EXCEPTION);
        assertThat(decision2.shouldRetry()).isTrue();
        assertThat(decision2.getBackoff()).isEqualTo(Duration.ofMillis(200));
        
        // Third attempt - should retry with doubled backoff again
        RetryPolicy.RetryDecision decision3 = policy.shouldRetry(3, TEST_EXCEPTION);
        assertThat(decision3.shouldRetry()).isTrue();
        assertThat(decision3.getBackoff()).isEqualTo(Duration.ofMillis(400));
        
        // Fourth attempt - should retry with doubled backoff again
        RetryPolicy.RetryDecision decision4 = policy.shouldRetry(4, TEST_EXCEPTION);
        assertThat(decision4.shouldRetry()).isTrue();
        assertThat(decision4.getBackoff()).isEqualTo(Duration.ofMillis(800));
        
        // Fifth attempt - should retry with max backoff (capped)
        RetryPolicy.RetryDecision decision5 = policy.shouldRetry(5, TEST_EXCEPTION);
        assertThat(decision5.shouldRetry()).isFalse();
        
        // Sixth attempt - should fail (max attempts reached)
        RetryPolicy.RetryDecision decision6 = policy.shouldRetry(6, TEST_EXCEPTION);
        assertThat(decision6.shouldRetry()).isFalse();
    }
    
    @Test
    void testExponentialBackoffWithJitterPolicy() {
        Duration initialBackoff = Duration.ofMillis(100);
        Duration maxBackoff = Duration.ofSeconds(1);
        int maxAttempts = 4;  // 4 attempts (1-indexed)
        double jitterFactor = 0.5;
        
        RetryPolicy policy = RetryPolicy.exponentialBackoffWithJitter(
                initialBackoff, maxAttempts, maxBackoff, jitterFactor);
        
        // Since we can't control random value, we check the range based on jitter factor
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, TEST_EXCEPTION);
        assertThat(decision1.shouldRetry()).isTrue();
        
        // Check that backoff values are within the expected range
        long minExpectedBackoff = (long)(initialBackoff.toMillis() * (1 - jitterFactor));
        long maxExpectedBackoff = (long)(initialBackoff.toMillis() * (1 + jitterFactor));
        assertThat(decision1.getBackoff().toMillis()).isBetween(minExpectedBackoff, maxExpectedBackoff);
        
        // Test with max attempts reached
        RetryPolicy.RetryDecision decision3 = policy.shouldRetry(3, TEST_EXCEPTION);
        assertThat(decision3.shouldRetry()).isTrue(); // Last valid attempt
        
        // This should be false since we've hit max attempts (3)
        RetryPolicy.RetryDecision decision4 = policy.shouldRetry(4, TEST_EXCEPTION);
        assertThat(decision4.shouldRetry()).isFalse();
    }
    
    @Test
    void testWithExceptionFilterPolicy() {
        RetryPolicy basePolicy = RetryPolicy.maxAttempts(3);
        
        // Filter that only retries RuntimeException
        RetryPolicy policy = RetryPolicy.withExceptionFilter(
                basePolicy, e -> e instanceof RuntimeException);
        
        // Should retry RuntimeException
        RetryPolicy.RetryDecision decision1 = policy.shouldRetry(1, new RuntimeException());
        assertThat(decision1.shouldRetry()).isTrue();
        
        // Should not retry other exceptions
        RetryPolicy.RetryDecision decision2 = policy.shouldRetry(1, new Exception());
        assertThat(decision2.shouldRetry()).isFalse();
    }
    
    @Test
    void testRetryDecisionFactory() {
        Duration backoff = Duration.ofMillis(100);
        
        RetryPolicy.RetryDecision retryDecision = RetryPolicy.RetryDecision.retry(backoff);
        assertThat(retryDecision.shouldRetry()).isTrue();
        assertThat(retryDecision.getBackoff()).isEqualTo(backoff);
        
        RetryPolicy.RetryDecision failDecision = RetryPolicy.RetryDecision.fail();
        assertThat(failDecision.shouldRetry()).isFalse();
        assertThat(failDecision.getBackoff()).isEqualTo(Duration.ZERO);
    }
}