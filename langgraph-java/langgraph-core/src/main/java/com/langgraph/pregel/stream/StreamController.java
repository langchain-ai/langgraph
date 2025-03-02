package com.langgraph.pregel.stream;

import com.langgraph.pregel.StreamMode;

import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Controls the streaming of results during Pregel execution.
 * Manages backpressure, cancellation, and output formatting.
 */
public class StreamController {
    private final Queue<Map<String, Object>> buffer;
    private final AtomicBoolean isCancelled;
    private final AtomicBoolean isPaused;
    private final Consumer<Map<String, Object>> outputConsumer;
    private final StreamMode streamMode;
    private int stepCount;
    
    /**
     * Create a StreamController.
     *
     * @param outputConsumer Consumer to receive output
     * @param streamMode Stream mode
     */
    public StreamController(Consumer<Map<String, Object>> outputConsumer, StreamMode streamMode) {
        this.buffer = new ConcurrentLinkedQueue<>();
        this.isCancelled = new AtomicBoolean(false);
        this.isPaused = new AtomicBoolean(false);
        this.outputConsumer = outputConsumer;
        this.streamMode = streamMode != null ? streamMode : StreamMode.VALUES;
        this.stepCount = 0;
    }
    
    /**
     * Process a new state update.
     *
     * @param state Current state
     * @param updatedChannels Set of channel names that were updated in this step
     * @param hasMoreWork Whether there is more work to do
     * @return True if execution should continue, false if it should stop
     */
    public boolean processUpdate(Map<String, Object> state, Set<String> updatedChannels, boolean hasMoreWork) {
        if (isCancelled.get()) {
            return false;
        }
        
        stepCount++;
        
        // Format output based on stream mode
        Map<String, Object> output = StreamOutput.format(
                state,
                updatedChannels,
                stepCount,
                hasMoreWork,
                streamMode
        );
        
        // Add to buffer and consume if not paused
        buffer.add(output);
        consumeOutput();
        
        return !isCancelled.get();
    }
    
    /**
     * Consume output from the buffer.
     */
    private void consumeOutput() {
        if (isPaused.get() || outputConsumer == null) {
            return;
        }
        
        Map<String, Object> output;
        while ((output = buffer.poll()) != null) {
            outputConsumer.accept(output);
            
            // Check if we should stop consuming
            if (isPaused.get() || isCancelled.get()) {
                break;
            }
        }
    }
    
    /**
     * Cancel streaming.
     */
    public void cancel() {
        isCancelled.set(true);
    }
    
    /**
     * Pause streaming.
     */
    public void pause() {
        isPaused.set(true);
    }
    
    /**
     * Resume streaming.
     */
    public void resume() {
        isPaused.set(false);
        consumeOutput();
    }
    
    /**
     * Check if streaming is cancelled.
     *
     * @return True if cancelled
     */
    public boolean isCancelled() {
        return isCancelled.get();
    }
    
    /**
     * Check if streaming is paused.
     *
     * @return True if paused
     */
    public boolean isPaused() {
        return isPaused.get();
    }
    
    /**
     * Get the current step count.
     *
     * @return Current step count
     */
    public int getStepCount() {
        return stepCount;
    }
    
    /**
     * Get the buffer size.
     *
     * @return Buffer size
     */
    public int getBufferSize() {
        return buffer.size();
    }
    
    /**
     * Get the stream mode.
     *
     * @return Stream mode
     */
    public StreamMode getStreamMode() {
        return streamMode;
    }
}