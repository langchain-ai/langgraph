package com.langgraph.pregel.registry;

import com.langgraph.pregel.PregelExecutable;
import com.langgraph.pregel.PregelNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class NodeRegistryTest {
    
    @Mock
    private PregelExecutable<Object, Object> mockAction;
    
    private PregelNode<Object, Object> mockNode1;
    private PregelNode<Object, Object> mockNode2;
    private PregelNode<Object, Object> mockNode3;
    
    @BeforeEach
    void setUp() {
        // Create real PregelNode instances with proper generic types
        mockNode1 = new PregelNode.Builder<>("node1", mockAction).build();
        mockNode2 = new PregelNode.Builder<>("node2", mockAction).build();
        mockNode3 = new PregelNode.Builder<>("node3", mockAction).build();
    }
    
    @Test
    void testEmptyRegistry() {
        NodeRegistry registry = new NodeRegistry();
        
        assertThat(registry.size()).isEqualTo(0);
        assertThat(registry.getAll()).isEmpty();
        assertThatThrownBy(() -> registry.get("nonexistent"))
                .isInstanceOf(NoSuchElementException.class)
                .hasMessageContaining("nonexistent");
    }
    
    @Test
    void testRegisterNode() {
        NodeRegistry registry = new NodeRegistry();
        
        registry.register(mockNode1);
        
        assertThat(registry.size()).isEqualTo(1);
        assertThat(registry.contains("node1")).isTrue();
        assertThat(registry.get("node1")).isSameAs(mockNode1);
    }
    
    @Test
    void testRegisterDuplicateNode() {
        NodeRegistry registry = new NodeRegistry();
        
        registry.register(mockNode1);
        
        assertThatThrownBy(() -> registry.register(mockNode1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("already registered");
        
        // Create new node with same name and proper generic type
        PregelNode<Object, Object> duplicateNode = new PregelNode.Builder<>("node1", mockAction).build();
        
        assertThatThrownBy(() -> registry.register(duplicateNode))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("already registered");
    }
    
    @Test
    void testRegisterAllNodes() {
        NodeRegistry registry = new NodeRegistry();
        
        registry.registerAll(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        assertThat(registry.size()).isEqualTo(3);
        assertThat(registry.contains("node1")).isTrue();
        assertThat(registry.contains("node2")).isTrue();
        assertThat(registry.contains("node3")).isTrue();
    }
    
    @Test
    void testConstructorWithCollection() {
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        assertThat(registry.size()).isEqualTo(2);
        assertThat(registry.contains("node1")).isTrue();
        assertThat(registry.contains("node2")).isTrue();
    }
    
    @Test
    void testConstructorWithMap() {
        Map<String, PregelNode<?, ?>> nodes = new HashMap<>();
        nodes.put("node1", mockNode1);
        nodes.put("node2", mockNode2);
        
        NodeRegistry registry = new NodeRegistry(nodes);
        
        assertThat(registry.size()).isEqualTo(2);
        assertThat(registry.contains("node1")).isTrue();
        assertThat(registry.contains("node2")).isTrue();
    }
    
    @Test
    void testConstructorWithMapNameMismatch() {
        Map<String, PregelNode<?, ?>> nodes = new HashMap<>();
        nodes.put("wrongName", mockNode1); // Node name is "node1" but map key is "wrongName"
        
        assertThatThrownBy(() -> new NodeRegistry(nodes))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Node name mismatch");
    }
    
    @Test
    void testRemoveNode() {
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        registry.remove("node1");
        
        assertThat(registry.size()).isEqualTo(1);
        assertThat(registry.contains("node1")).isFalse();
        assertThat(registry.contains("node2")).isTrue();
    }
    
    @Test
    void testGetSubscribers() {
        // Use PregelNode.Builder to add subscriptions with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .channels("channel1")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .channels("channel2")
            .build();
        
        mockNode3 = new PregelNode.Builder<Object, Object>("node3", mockAction)
            .channels("channel1")
            .channels("channel2")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode<?, ?>> channel1Subscribers = registry.getSubscribers("channel1");
        Set<PregelNode<?, ?>> channel2Subscribers = registry.getSubscribers("channel2");
        
        assertThat(channel1Subscribers).containsExactlyInAnyOrder(mockNode1, mockNode3);
        assertThat(channel2Subscribers).containsExactlyInAnyOrder(mockNode2, mockNode3);
    }
    
    @Test
    void testGetTriggered() {
        // Use PregelNode.Builder to set triggers with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .triggerChannels("trigger1")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .triggerChannels("trigger2")
            .build();
        
        mockNode3 = new PregelNode.Builder<Object, Object>("node3", mockAction)
            .triggerChannels("trigger1")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode<?, ?>> trigger1Nodes = registry.getTriggered("trigger1");
        Set<PregelNode<?, ?>> trigger2Nodes = registry.getTriggered("trigger2");
        
        assertThat(trigger1Nodes).containsExactlyInAnyOrder(mockNode1, mockNode3);
        assertThat(trigger2Nodes).containsExactlyInAnyOrder(mockNode2);
    }
    
    @Test
    void testGetWriters() {
        // Use PregelNode.Builder to set writers with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .writers("channel1")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .writers("channel2")
            .build();
        
        mockNode3 = new PregelNode.Builder<Object, Object>("node3", mockAction)
            .writers("channel1")
            .writers("channel2")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode<?, ?>> channel1Writers = registry.getWriters("channel1");
        Set<PregelNode<?, ?>> channel2Writers = registry.getWriters("channel2");
        
        assertThat(channel1Writers).containsExactlyInAnyOrder(mockNode1, mockNode3);
        assertThat(channel2Writers).containsExactlyInAnyOrder(mockNode2, mockNode3);
    }
    
    @Test
    void testValidateSuccess() {
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        // Should not throw any exceptions
        registry.validate();
    }
    
    @Test
    void testValidateSubscriptionsFail() {
        // Use PregelNode.Builder to set subscriptions with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .channels("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .channels("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateSubscriptions(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("reads from non-existent channel");
    }
    
    @Test
    void testValidateWritersFail() {
        // Use PregelNode.Builder to set writers with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .writers("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .writers("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateWriters(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("writes to non-existent channel");
    }
    
    @Test
    void testValidateTriggersFail() {
        // Use PregelNode.Builder to set triggers with generic types
        mockNode1 = new PregelNode.Builder<Object, Object>("node1", mockAction)
            .triggerChannels("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder<Object, Object>("node2", mockAction)
            .triggerChannels("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateTriggers(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("has non-existent trigger");
    }
}