# Java Channel Interfaces

This document defines the Java interfaces for the channel system of LangGraph, closely aligned with the Python implementation.

## `Channel` Interface

The base interface for all channels, providing methods for getting values, applying updates, and checkpoint management.

```java
package com.langgraph.channels;

/**
 * Interface for communication channels between nodes in a graph.
 */
public interface Channel {
    /**
     * Get the current value of the channel.
     *
     * @return Current value
     */
    Object getValue();
    
    /**
     * Update the channel with a new value.
     *
     * @param value New value
     * @return True if the update was applied, false otherwise
     */
    boolean update(Object value);
    
    /**
     * Get the value to save in a checkpoint.
     *
     * @return Checkpointed value
     */
    Object checkpoint();
    
    /**
     * Restore the channel from a checkpoint.
     *
     * @param value Checkpointed value
     */
    void fromCheckpoint(Object value);
}
```

## Channel Implementations

### `LastValue` Channel

A channel that stores a single value, replacing it with each update.

```java
package com.langgraph.channels;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Channel that stores the last value set, rejecting multiple updates within a single step.
 * This is the default channel type for most use cases.
 */
public class LastValue implements Channel {
    private final AtomicReference<Object> value = new AtomicReference<>();
    private boolean updated = false;
    
    /**
     * Create a LastValue channel with an optional initial value.
     *
     * @param initialValue Optional initial value
     */
    public LastValue(Object initialValue) {
        value.set(initialValue);
    }
    
    /**
     * Create an empty LastValue channel.
     */
    public LastValue() {
        this(null);
    }
    
    @Override
    public Object getValue() {
        return value.get();
    }
    
    @Override
    public boolean update(Object newValue) {
        if (updated) {
            throw new IllegalStateException("LastValue channel cannot be updated multiple times in one step");
        }
        
        // Skip update if value hasn't changed
        if (Objects.equals(value.get(), newValue)) {
            return false;
        }
        
        value.set(newValue);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return value.get();
    }
    
    @Override
    public void fromCheckpoint(Object checkpointValue) {
        value.set(checkpointValue);
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
}
```

### `AnyValue` Channel

A channel that accepts multiple updates within a step, storing only the last one.

```java
package com.langgraph.channels;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Channel that accepts multiple updates within a step, storing only the last one.
 */
public class AnyValue implements Channel {
    private final AtomicReference<Object> value = new AtomicReference<>();
    private boolean updated = false;
    
    /**
     * Create an AnyValue channel with an optional initial value.
     *
     * @param initialValue Optional initial value
     */
    public AnyValue(Object initialValue) {
        value.set(initialValue);
    }
    
    /**
     * Create an empty AnyValue channel.
     */
    public AnyValue() {
        this(null);
    }
    
    @Override
    public Object getValue() {
        return value.get();
    }
    
    @Override
    public boolean update(Object newValue) {
        // Skip update if value hasn't changed
        if (Objects.equals(value.get(), newValue)) {
            return false;
        }
        
        value.set(newValue);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return value.get();
    }
    
    @Override
    public void fromCheckpoint(Object checkpointValue) {
        value.set(checkpointValue);
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
}
```

### `EphemeralValue` Channel

A channel that clears its value after being read.

```java
package com.langgraph.channels;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Channel that clears its value after being read.
 * Useful for temporary values that should only be processed once.
 */
public class EphemeralValue implements Channel {
    private final AtomicReference<Object> value = new AtomicReference<>();
    private boolean updated = false;
    
    /**
     * Create an EphemeralValue channel with an optional initial value.
     *
     * @param initialValue Optional initial value
     */
    public EphemeralValue(Object initialValue) {
        value.set(initialValue);
    }
    
    /**
     * Create an empty EphemeralValue channel.
     */
    public EphemeralValue() {
        this(null);
    }
    
    @Override
    public Object getValue() {
        Object currentValue = value.getAndSet(null);
        return currentValue;
    }
    
    @Override
    public boolean update(Object newValue) {
        // Skip update if value hasn't changed
        if (Objects.equals(value.get(), newValue)) {
            return false;
        }
        
        value.set(newValue);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return value.get();
    }
    
    @Override
    public void fromCheckpoint(Object checkpointValue) {
        value.set(checkpointValue);
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
}
```

### `UntrackedValue` Channel

A channel that's excluded from checkpoints.

```java
package com.langgraph.channels;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Channel that's excluded from checkpoints.
 * Useful for storing large temporary data that shouldn't be persisted.
 */
public class UntrackedValue implements Channel {
    private final AtomicReference<Object> value = new AtomicReference<>();
    private boolean updated = false;
    
    /**
     * Create an UntrackedValue channel with an optional initial value.
     *
     * @param initialValue Optional initial value
     */
    public UntrackedValue(Object initialValue) {
        value.set(initialValue);
    }
    
    /**
     * Create an empty UntrackedValue channel.
     */
    public UntrackedValue() {
        this(null);
    }
    
    @Override
    public Object getValue() {
        return value.get();
    }
    
    @Override
    public boolean update(Object newValue) {
        // Skip update if value hasn't changed
        if (Objects.equals(value.get(), newValue)) {
            return false;
        }
        
        value.set(newValue);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        // Return null for checkpoint as this value is not tracked
        return null;
    }
    
    @Override
    public void fromCheckpoint(Object checkpointValue) {
        // No-op as this channel isn't tracked in checkpoints
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
}
```

### `Topic` Channel

A publish-subscribe channel supporting multiple values and subscribers.

```java
package com.langgraph.channels;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Publish-subscribe channel supporting multiple values and subscribers.
 * Each subscriber receives all published values.
 */
public class Topic implements Channel {
    private final List<Object> values = new CopyOnWriteArrayList<>();
    private boolean updated = false;
    
    @Override
    public Object getValue() {
        List<Object> result = new ArrayList<>(values);
        values.clear();
        return result;
    }
    
    @Override
    public boolean update(Object value) {
        values.add(value);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return new ArrayList<>(values);
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public void fromCheckpoint(Object checkpointValue) {
        values.clear();
        if (checkpointValue != null) {
            values.addAll((List<Object>) checkpointValue);
        }
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
}
```

### `BinaryOperatorAggregate` Channel

A channel that aggregates values using a binary operator.

```java
package com.langgraph.channels;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BinaryOperator;

/**
 * Channel that aggregates values using a binary operator.
 *
 * @param <T> Type of values to aggregate
 */
public class BinaryOperatorAggregate<T> implements Channel {
    private final AtomicReference<T> value = new AtomicReference<>();
    private final BinaryOperator<T> operator;
    private boolean updated = false;
    
    /**
     * Create a BinaryOperatorAggregate channel with an operator and optional initial value.
     *
     * @param operator Binary operator for combining values
     * @param initialValue Optional initial value
     */
    public BinaryOperatorAggregate(BinaryOperator<T> operator, T initialValue) {
        this.operator = operator;
        value.set(initialValue);
    }
    
    /**
     * Create a BinaryOperatorAggregate channel with an operator.
     *
     * @param operator Binary operator for combining values
     */
    public BinaryOperatorAggregate(BinaryOperator<T> operator) {
        this(operator, null);
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public T getValue() {
        return value.get();
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public boolean update(Object newValue) {
        T typedValue = (T) newValue;
        T currentValue = value.get();
        
        if (currentValue == null) {
            value.set(typedValue);
            updated = true;
            return true;
        }
        
        // Apply the binary operator to combine values
        T combinedValue = operator.apply(currentValue, typedValue);
        
        // Skip update if value hasn't changed
        if (Objects.equals(currentValue, combinedValue)) {
            return false;
        }
        
        value.set(combinedValue);
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return value.get();
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public void fromCheckpoint(Object checkpointValue) {
        value.set((T) checkpointValue);
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
    
    /**
     * Factory for creating sum aggregates.
     *
     * @param <T> Type of values to sum
     * @return A channel that sums values
     */
    public static <T extends Number> BinaryOperatorAggregate<T> sum() {
        return new BinaryOperatorAggregate<>((a, b) -> {
            if (a instanceof Integer) {
                return (T) Integer.valueOf(((Integer) a) + ((Integer) b));
            } else if (a instanceof Long) {
                return (T) Long.valueOf(((Long) a) + ((Long) b));
            } else if (a instanceof Double) {
                return (T) Double.valueOf(((Double) a) + ((Double) b));
            } else if (a instanceof Float) {
                return (T) Float.valueOf(((Float) a) + ((Float) b));
            } else {
                throw new IllegalArgumentException("Unsupported number type: " + a.getClass());
            }
        });
    }
}
```

### `NamedBarrierValue` Channel

A synchronization mechanism requiring all named values to be received.

```java
package com.langgraph.channels;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Synchronization channel requiring all named values to be received.
 * Triggers when all expected names have provided values.
 */
public class NamedBarrierValue implements Channel {
    private final Set<String> expectedNames;
    private final Map<String, Object> values = new ConcurrentHashMap<>();
    private boolean updated = false;
    
    /**
     * Create a NamedBarrierValue channel with expected names.
     *
     * @param expectedNames Set of names expected to provide values
     */
    public NamedBarrierValue(Set<String> expectedNames) {
        this.expectedNames = new HashSet<>(expectedNames);
    }
    
    @Override
    public Object getValue() {
        // Return the map of collected values if all expected names have provided values
        if (values.keySet().containsAll(expectedNames)) {
            Map<String, Object> result = new HashMap<>(values);
            values.clear();
            return result;
        }
        
        // Return null if the barrier is not satisfied
        return null;
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public boolean update(Object value) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("NamedBarrierValue requires a Map<String, Object> update");
        }
        
        Map<String, Object> update = (Map<String, Object>) value;
        if (update.size() != 1) {
            throw new IllegalArgumentException("NamedBarrierValue update must contain exactly one entry");
        }
        
        String name = update.keySet().iterator().next();
        if (!expectedNames.contains(name)) {
            throw new IllegalArgumentException("Unexpected name in NamedBarrierValue update: " + name);
        }
        
        values.put(name, update.get(name));
        updated = true;
        return true;
    }
    
    @Override
    public Object checkpoint() {
        return new HashMap<>(values);
    }
    
    @Override
    @SuppressWarnings("unchecked")
    public void fromCheckpoint(Object checkpointValue) {
        values.clear();
        if (checkpointValue != null) {
            values.putAll((Map<String, Object>) checkpointValue);
        }
        updated = false;
    }
    
    /**
     * Reset the update flag at the end of a superstep.
     */
    public void resetUpdated() {
        updated = false;
    }
    
    /**
     * Check if the channel was updated in the current step.
     *
     * @return True if updated, false otherwise
     */
    public boolean wasUpdated() {
        return updated;
    }
    
    /**
     * Check if all expected names have provided values.
     *
     * @return True if the barrier is satisfied, false otherwise
     */
    public boolean isBarrierSatisfied() {
        return values.keySet().containsAll(expectedNames);
    }
}
```

## Channel Factory

A factory class for creating channels.

```java
package com.langgraph.channels;

import java.util.Set;
import java.util.function.BinaryOperator;

/**
 * Factory for creating channels.
 */
public final class Channels {
    private Channels() {}
    
    /**
     * Create a LastValue channel.
     *
     * @param initialValue Optional initial value
     * @return LastValue channel
     */
    public static LastValue lastValue(Object initialValue) {
        return new LastValue(initialValue);
    }
    
    /**
     * Create an empty LastValue channel.
     *
     * @return LastValue channel
     */
    public static LastValue lastValue() {
        return new LastValue();
    }
    
    /**
     * Create an AnyValue channel.
     *
     * @param initialValue Optional initial value
     * @return AnyValue channel
     */
    public static AnyValue anyValue(Object initialValue) {
        return new AnyValue(initialValue);
    }
    
    /**
     * Create an empty AnyValue channel.
     *
     * @return AnyValue channel
     */
    public static AnyValue anyValue() {
        return new AnyValue();
    }
    
    /**
     * Create an EphemeralValue channel.
     *
     * @param initialValue Optional initial value
     * @return EphemeralValue channel
     */
    public static EphemeralValue ephemeralValue(Object initialValue) {
        return new EphemeralValue(initialValue);
    }
    
    /**
     * Create an empty EphemeralValue channel.
     *
     * @return EphemeralValue channel
     */
    public static EphemeralValue ephemeralValue() {
        return new EphemeralValue();
    }
    
    /**
     * Create an UntrackedValue channel.
     *
     * @param initialValue Optional initial value
     * @return UntrackedValue channel
     */
    public static UntrackedValue untrackedValue(Object initialValue) {
        return new UntrackedValue(initialValue);
    }
    
    /**
     * Create an empty UntrackedValue channel.
     *
     * @return UntrackedValue channel
     */
    public static UntrackedValue untrackedValue() {
        return new UntrackedValue();
    }
    
    /**
     * Create a Topic channel.
     *
     * @return Topic channel
     */
    public static Topic topic() {
        return new Topic();
    }
    
    /**
     * Create a BinaryOperatorAggregate channel.
     *
     * @param operator Binary operator for combining values
     * @param initialValue Optional initial value
     * @param <T> Type of values to aggregate
     * @return BinaryOperatorAggregate channel
     */
    public static <T> BinaryOperatorAggregate<T> binaryOperatorAggregate(
            BinaryOperator<T> operator, T initialValue) {
        return new BinaryOperatorAggregate<>(operator, initialValue);
    }
    
    /**
     * Create a BinaryOperatorAggregate channel.
     *
     * @param operator Binary operator for combining values
     * @param <T> Type of values to aggregate
     * @return BinaryOperatorAggregate channel
     */
    public static <T> BinaryOperatorAggregate<T> binaryOperatorAggregate(BinaryOperator<T> operator) {
        return new BinaryOperatorAggregate<>(operator);
    }
    
    /**
     * Create a NamedBarrierValue channel.
     *
     * @param expectedNames Set of names expected to provide values
     * @return NamedBarrierValue channel
     */
    public static NamedBarrierValue namedBarrierValue(Set<String> expectedNames) {
        return new NamedBarrierValue(expectedNames);
    }
}
```