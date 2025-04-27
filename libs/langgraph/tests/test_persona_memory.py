"""
Tests for the PersonaMemory class.

This module contains tests for the PersonaMemory class, which handles memory
management for persona-based interactions. It includes tests for trait extraction,
entity recognition, and memory updates.
"""

import pytest
from langgraph.experimental.memory.persona_memory import PersonaMemory
import spacy


@pytest.fixture
def persona_memory():
    """
    Creates a fresh PersonaMemory instance for testing.

    Returns:
        PersonaMemory: A new instance of PersonaMemory.
    """
    return PersonaMemory()


def test_extract_traits_basic_emotions(persona_memory):
    """
    Tests basic emotion trait extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Positive emotions are correctly identified and mapped
        - Negative emotions are correctly identified and mapped
    """
    # Test positive emotions
    text = "I am so happy and excited to see you!"
    traits = persona_memory.extract_traits(text)
    assert "positive" in traits
    assert len(traits) == 1  # Both map to same trait

    # Test negative emotions
    text = "I feel sad and angry about the situation."
    traits = persona_memory.extract_traits(text)
    assert "negative" in traits
    assert len(traits) == 1  # Both map to same trait


def test_extract_traits_personality_traits(persona_memory):
    """
    Tests personality trait extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Analytical traits are correctly identified
        - Friendly traits are correctly identified
    """
    # Test analytical traits
    text = "She is very analytical and logical in her approach."
    traits = persona_memory.extract_traits(text)
    assert "analytical" in traits
    assert len(traits) == 1  # Both map to same trait

    # Test friendly traits
    text = "He is friendly and kind to everyone."
    traits = persona_memory.extract_traits(text)
    assert "friendly" in traits
    assert len(traits) == 1  # Both map to same trait


def test_extract_traits_cautious_traits(persona_memory):
    """
    Tests cautious trait extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that cautious traits are correctly identified and mapped.
    """
    text = "She was nervous and hesitant about the decision."
    traits = persona_memory.extract_traits(text)
    assert "cautious" in traits
    assert len(traits) == 1  # Both map to same trait


def test_extract_traits_courageous_traits(persona_memory):
    """
    Tests courageous trait extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that courageous traits are correctly identified and mapped.
    """
    text = "He showed brave and fearless behavior."
    traits = persona_memory.extract_traits(text)
    assert "courageous" in traits
    assert len(traits) == 1  # Both map to same trait


def test_extract_traits_no_traits(persona_memory):
    """
    Tests behavior when no traits are present in the text.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Empty trait list is returned when no traits are present
        - System handles neutral text appropriately
    """
    text = "The rock sat still on the ground."
    traits = persona_memory.extract_traits(text)
    assert traits == []


def test_extract_entities_people(persona_memory):
    """
    Tests people entity extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - People names are correctly identified
        - Multiple people are properly extracted
        - Entity counts are accurate
    """
    text = "John and Mary visited the museum."
    entities = persona_memory.extract_entities(text)
    assert "John" in entities["people"]
    assert "Mary" in entities["people"]
    assert len(entities["people"]) == 2


def test_extract_entities_places(persona_memory):
    """
    Tests place entity extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Place names are correctly identified
        - Multiple places are properly extracted
        - Entity counts are accurate
    """
    text = "They traveled to Paris and London."
    entities = persona_memory.extract_entities(text)
    assert "Paris" in entities["places"]
    assert "London" in entities["places"]
    assert len(entities["places"]) == 2


def test_extract_entities_organizations(persona_memory):
    """
    Tests organization entity extraction functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Organization names are correctly identified
        - Multiple organizations are properly extracted
        - Entity counts are accurate
    """
    text = "She works at Google and Microsoft."
    entities = persona_memory.extract_entities(text)
    assert "Google" in entities["places"]
    assert "Microsoft" in entities["places"]
    assert len(entities["places"]) == 2


def test_update_memory_traits(persona_memory):
    """
    Tests updating memory with new traits.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - New traits are correctly added to memory
        - Multiple traits can be added sequentially
        - Trait counts are accurate
    """
    text = "I am happy and excited about the project."
    persona_memory.update_memory(text)
    assert "positive" in persona_memory.traits
    assert len(persona_memory.traits) == 1

    # Add more traits
    text = "I am also analytical and logical."
    persona_memory.update_memory(text)
    assert "analytical" in persona_memory.traits
    assert len(persona_memory.traits) == 2


def test_update_memory_associations(persona_memory):
    """
    Tests updating memory with new entity associations.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Person-entity associations are correctly stored
        - Place-entity associations are correctly stored
        - Traits are properly associated with entities
    """
    text = "John is happy and excited about Paris."
    persona_memory.update_memory(text)

    # Check person association
    assert "John" in persona_memory.associations
    assert persona_memory.associations["John"].entity_type == "person"
    assert "positive" in persona_memory.associations["John"].traits

    # Check place association
    assert "Paris" in persona_memory.associations
    assert persona_memory.associations["Paris"].entity_type == "place"
    assert "positive" in persona_memory.associations["Paris"].traits


def test_update_memory_duplicate_traits(persona_memory):
    """
    Tests handling of duplicate traits in memory updates.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Duplicate traits are properly deduplicated
        - Memory state remains consistent with duplicates
    """
    text = "I am happy and happy and happy."
    persona_memory.update_memory(text)
    assert "positive" in persona_memory.traits
    assert len(persona_memory.traits) == 1  # Duplicates removed


def test_get_summary(persona_memory):
    """
    Tests the memory summary generation functionality.

    Args:
        persona_memory (PersonaMemory): The PersonaMemory instance to test.

    This test verifies that:
        - Summary contains all required sections
        - Entity associations are properly included
        - Trait associations are correctly represented
    """
    # Add some data
    text = "John is happy about Paris."
    persona_memory.update_memory(text)

    summary = persona_memory.get_summary()
    assert "traits" in summary
    assert "associations" in summary
    assert "John" in summary["associations"]
    assert "Paris" in summary["associations"]
    assert summary["associations"]["John"]["type"] == "person"
    assert summary["associations"]["Paris"]["type"] == "place"
    assert "positive" in summary["associations"]["John"]["traits"]
    assert "positive" in summary["associations"]["Paris"]["traits"]


def simulate_api_response(content: str) -> str:
    """
    Simulates an API response format (e.g., OpenAI chat completions).

    Args:
        content (str): The content to simulate as an assistant's message.

    Returns:
        str: Extracted assistant message content.
    """
    simulated_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }
        ]
    }
    return simulated_response["choices"][0]["message"]["content"]


def test_persona_memory_full_conversation_pipeline(persona_memory):
    """
    Simulates a full conversation with persona memory updates.

    This test simulates both user prompts and assistant API responses over time,
    verifying that traits and associations accumulate correctly across a session.
    """
    
    conversation_flow = [
        {
            "user_input": "Hi there! I'm thinking of visiting Paris.",
            "assistant_output": "That's fantastic! Paris is full of excitement and beautiful sights."
        },
        {
            "user_input": "I'm a bit nervous traveling alone though.",
            "assistant_output": "That's understandable — feeling a little nervous shows you're thoughtful and careful."
        },
        {
            "user_input": "Thank you! I'm feeling braver now.",
            "assistant_output": "Wonderful to hear! You are courageous and positive."
        },
        {
            "user_input": "My friend John is joining me in London.",
            "assistant_output": "London is another amazing place! And having John with you will be great company."
        }
    ]

    for step in conversation_flow:
        # Normally user input could also be processed, but for now we only process assistant output
        assistant_message = simulate_api_response(step["assistant_output"])
        persona_memory.update_memory(assistant_message)

    summary = persona_memory.get_summary()

    # 1. Traits that should have been detected over conversation
    expected_traits = {"positive", "cautious", "courageous", "friendly", "analytical"}
    detected_traits = set(summary["traits"])

    # We expect at least some (not necessarily all) traits to appear
    assert "positive" in detected_traits
    assert "cautious" in detected_traits
    assert "courageous" in detected_traits

    # 2. Entity associations
    associations = summary["associations"]
    
    assert "Paris" in associations
    assert associations["Paris"]["type"] == "place"
    assert "positive" in associations["Paris"]["traits"]

    assert "London" in associations
    assert associations["London"]["type"] == "place"

    assert "John" in associations
    assert associations["John"]["type"] == "person"

    # 3. Integrity checks
    for entity_data in associations.values():
        assert isinstance(entity_data["traits"], list)

    assert isinstance(summary["traits"], list)