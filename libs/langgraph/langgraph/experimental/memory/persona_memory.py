"""Persona memory module.

Tracks evolving persona traits, emotions, and associations with people, places, and events across a session.
Part of langgraph.experimental.memory.
"""

from typing import Any, Dict, List, Optional

import spacy
from pydantic import BaseModel, Field

# Load the small English language model for NLP processing
nlp = spacy.load("en_core_web_sm")


class AssociatedEntity(BaseModel):
    """Represents a person, place, or event associated with traits or emotions."""

    name: str
    entity_type: str  # "person", "place", "event"
    traits: List[str] = Field(default_factory=list)
    emotions: List[str] = Field(default_factory=list)


class PersonaMemory(BaseModel):
    """Memory module for tracking persona traits and associations."""

    traits: List[str] = Field(default_factory=list)
    associations: Dict[str, AssociatedEntity] = Field(default_factory=dict)

    def extract_traits(self, text: str) -> List[str]:
        """Extract traits from text based on detected adjectives.

        Extract traits based on adjective matching using a predefined synonym-to-trait mapping.

        Future improvements:
        - Integrate external synonym databases (e.g., WordNet) for richer matching.
        - Use spaCy word vectors (`similarity()`) to detect fuzzy matches.
        - Fine-tune thresholds based on domain-specific needs (e.g., agent conversations).
        """

        # Mapping from adjectives to high-level persona traits.
        # Each adjective maps to a single trait to avoid inflating profiles.
        # In the future, multi-trait activation could be added for richer modeling.
        # Alternatively, we could use a more sophisticated approach like a decision tree or a neural network.

        synonym_to_trait = {
            # Positive / enthusiastic
            "happy": "positive",
            "joyful": "positive",
            "cheerful": "positive",
            "excited": "positive",
            "enthusiastic": "positive",
            "delighted": "positive",
            "thrilled": "positive",
            "amused": "positive",
            "fantastic": "positive",
            "wonderful": "positive",
            "amazing": "positive",
            "great": "positive",
            "beautiful": "positive",
            # Negative / cautious
            "sad": "negative",
            "angry": "negative",
            "upset": "negative",
            "fearful": "cautious",
            "nervous": "cautious",
            "hesitant": "cautious",
            "anxious": "cautious",
            "worried": "cautious",
            "uncertain": "cautious",
            "doubtful": "cautious",
            # Courageous / brave
            "brave": "courageous",
            "bold": "courageous",
            "fearless": "courageous",
            "heroic": "courageous",
            "daring": "courageous",
            "courageous": "courageous",
            # Friendly / welcoming
            "friendly": "friendly",
            "kind": "friendly",
            "pleasant": "friendly",
            "charming": "friendly",
            "polite": "friendly",
            # Analytical / logical
            "analytical": "analytical",
            "logical": "analytical",
            "rational": "analytical",
            "thoughtful": "analytical",
            "meticulous": "analytical",
            "curious": "analytical",
        }

        traits_detected = set()

        # Process text with spaCy
        doc = nlp(text)  # nlp already loaded earlier

        for token in doc:
            if token.pos_ == "ADJ":
                lemma = token.lemma_.lower()
                if lemma in synonym_to_trait:
                    trait = synonym_to_trait[lemma]
                    traits_detected.add(trait)

        return list(traits_detected)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities (people and places) from text."""
        doc = nlp(text)
        people = []
        places = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.append(ent.text)
            elif ent.label_ in {"GPE", "LOC", "ORG", "EVENT"}:
                places.append(ent.text)

        return {"people": list(set(people)), "places": list(set(places))}

    def update_memory(self, text: str) -> None:
        """Update persona memory based on new text."""
        # Extract traits and entities
        traits = self.extract_traits(text)
        entities = self.extract_entities(text)

        # Update traits
        self.traits.extend(traits)
        self.traits = list(set(self.traits))  # Remove duplicates

        # Update associations
        for person in entities["people"]:
            if person not in self.associations:
                self.associations[person] = AssociatedEntity(
                    name=person, entity_type="person", traits=traits
                )
            else:
                self.associations[person].traits.extend(traits)
                self.associations[person].traits = list(
                    set(self.associations[person].traits)
                )

        for place in entities["places"]:
            if place not in self.associations:
                self.associations[place] = AssociatedEntity(
                    name=place, entity_type="place", traits=traits
                )
            else:
                self.associations[place].traits.extend(traits)
                self.associations[place].traits = list(
                    set(self.associations[place].traits)
                )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of current persona traits and associations."""
        return {
            "traits": self.traits,
            "associations": {
                name: {"type": entity.entity_type, "traits": entity.traits}
                for name, entity in self.associations.items()
            },
        }
