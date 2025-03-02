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
    private PregelExecutable mockAction;
    
    private PregelNode mockNode1;
    private PregelNode mockNode2;
    private PregelNode mockNode3;
    
    @BeforeEach
    void setUp() {
        // Create real PregelNode instances instead of mocks
        mockNode1 = new PregelNode("node1", mockAction);
        mockNode2 = new PregelNode("node2", mockAction);
        mockNode3 = new PregelNode("node3", mockAction);
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
        
        // Create new node with same name
        PregelNode duplicateNode = new PregelNode("node1", mockAction);
        
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
        Map<String, PregelNode> nodes = new HashMap<>();
        nodes.put("node1", mockNode1);
        nodes.put("node2", mockNode2);
        
        NodeRegistry registry = new NodeRegistry(nodes);
        
        assertThat(registry.size()).isEqualTo(2);
        assertThat(registry.contains("node1")).isTrue();
        assertThat(registry.contains("node2")).isTrue();
    }
    
    @Test
    void testConstructorWithMapNameMismatch() {
        Map<String, PregelNode> nodes = new HashMap<>();
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
        // Use PregelNode.Builder to add subscriptions
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .subscribe("channel1")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .subscribe("channel2")
            .build();
        
        mockNode3 = new PregelNode.Builder("node3", mockAction)
            .subscribe("channel1")
            .subscribe("channel2")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode> channel1Subscribers = registry.getSubscribers("channel1");
        Set<PregelNode> channel2Subscribers = registry.getSubscribers("channel2");
        
        assertThat(channel1Subscribers).containsExactlyInAnyOrder(mockNode1, mockNode3);
        assertThat(channel2Subscribers).containsExactlyInAnyOrder(mockNode2, mockNode3);
    }
    
    @Test
    void testGetTriggered() {
        // Use PregelNode.Builder to set triggers
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .trigger("trigger1")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .trigger("trigger2")
            .build();
        
        mockNode3 = new PregelNode.Builder("node3", mockAction)
            .trigger("trigger1")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode> trigger1Nodes = registry.getTriggered("trigger1");
        Set<PregelNode> trigger2Nodes = registry.getTriggered("trigger2");
        
        assertThat(trigger1Nodes).containsExactlyInAnyOrder(mockNode1, mockNode3);
        assertThat(trigger2Nodes).containsExactlyInAnyOrder(mockNode2);
    }
    
    @Test
    void testGetWriters() {
        // Use PregelNode.Builder to set writers
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .writer("channel1")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .writer("channel2")
            .build();
        
        mockNode3 = new PregelNode.Builder("node3", mockAction)
            .writer("channel1")
            .writer("channel2")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2, mockNode3));
        
        Set<PregelNode> channel1Writers = registry.getWriters("channel1");
        Set<PregelNode> channel2Writers = registry.getWriters("channel2");
        
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
        // Use PregelNode.Builder to set subscriptions
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .subscribe("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .subscribe("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateSubscriptions(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("subscribes to non-existent channel");
    }
    
    @Test
    void testValidateWritersFail() {
        // Use PregelNode.Builder to set writers
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .writer("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .writer("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateWriters(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("writes to non-existent channel");
    }
    
    @Test
    void testValidateTriggersFail() {
        // Use PregelNode.Builder to set triggers
        mockNode1 = new PregelNode.Builder("node1", mockAction)
            .trigger("validChannel")
            .build();
        
        mockNode2 = new PregelNode.Builder("node2", mockAction)
            .trigger("invalidChannel")
            .build();
        
        NodeRegistry registry = new NodeRegistry(Arrays.asList(mockNode1, mockNode2));
        
        Set<String> validChannels = Collections.singleton("validChannel");
        
        assertThatThrownBy(() -> registry.validateTriggers(validChannels))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("has non-existent trigger");
    }
}