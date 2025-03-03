package com.langgraph.channels;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

/**
 * A runtime type token for preserving generic type information.
 * Used to capture generic type parameters that would otherwise be erased.
 * 
 * <p>Usage example:
 * <pre>
 * TypeReference&lt;List&lt;String&gt;&gt; listStringType = new TypeReference&lt;List&lt;String&gt;&gt;() {};
 * </pre>
 * 
 * @param <T> The type to capture
 */
public abstract class TypeReference<T> {
    private final Type type;
    
    /**
     * Creates a new type reference, capturing the generic type parameter T.
     * Due to Java's type erasure, this constructor must be called from an
     * anonymous subclass to capture the type information.
     */
    protected TypeReference() {
        Type superclass = getClass().getGenericSuperclass();
        if (superclass instanceof ParameterizedType) {
            // Extract the actual type argument from the anonymous subclass
            type = ((ParameterizedType) superclass).getActualTypeArguments()[0];
        } else {
            throw new IllegalArgumentException("TypeReference must be created with type parameters");
        }
    }
    
    /**
     * Gets the captured type.
     * 
     * @return The captured Type
     */
    public Type getType() {
        return type;
    }
    
    /**
     * Returns the raw Class for this type reference.
     * 
     * @return The raw Class
     */
    @SuppressWarnings("unchecked")
    public Class<T> getRawClass() {
        try {
            if (type instanceof Class<?>) {
                return (Class<T>) type;
            } else if (type instanceof ParameterizedType) {
                return (Class<T>) ((ParameterizedType) type).getRawType();
            } else {
                // Handle type variables (like T) by returning Object.class as a fallback
                return (Class<T>) Object.class;
            }
        } catch (Exception e) {
            // If we encounter any other issue, fallback to Object.class
            return (Class<T>) Object.class;
        }
    }
    
    @Override
    public String toString() {
        return "TypeReference<" + type + ">";
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass().getSuperclass() != o.getClass().getSuperclass()) return false;
        
        TypeReference<?> that = (TypeReference<?>) o;
        return type.equals(that.type);
    }
    
    @Override
    public int hashCode() {
        return type.hashCode();
    }
}