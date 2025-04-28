"""
Persona memory module.

Tracks evolving persona traits, emotions, and associations with people, places, and events across a session.
Part of langgraph.experimental.memory.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar

import spacy
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from langgraph.checkpoint.base import BaseCheckpointSaver

# Load the small English language model for NLP processing
nlp = spacy.load("en_core_web_sm")

V = TypeVar("V", int, float, str)


class AssociatedEntity(BaseModel):
    """Represents a person, place, or event associated with traits or emotions."""

    name: str
    entity_type: str  # "person", "place", "event"
    traits: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)


class PersonaMemorySnapshot(BaseModel):
    """Serializable snapshot of PersonaMemory state."""

    traits: list[str]
    associations: dict[str, AssociatedEntity]


class PersonaMemory(BaseModel):
    """Memory module for tracking persona traits and associations."""

    traits: list[str] = Field(default_factory=list)
    associations: dict[str, AssociatedEntity] = Field(default_factory=dict)

    def extract_traits(
        self, text: str, extra_synonyms: Optional[dict[str, str]] = None
    ) -> list[str]:
        """Extract traits from text based on detected adjectives.

        Extract traits based on adjective matching using a predefined synonym-to-trait mapping.

        Args:
            text: The input text to analyze
            extra_synonyms: Optional dictionary of additional synonyms to trait mappings

        Future improvements:
        - Integrate external synonym databases (e.g., WordNet) for richer matching.
        - Use spaCy word vectors (`similarity()`) to detect fuzzy matches.
        - Fine-tune thresholds based on domain-specific needs (e.g., agent conversations).
        """
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

        # Merge extra synonyms if provided
        if extra_synonyms is not None:
            synonym_to_trait.update(extra_synonyms)

        traits_detected = set()
        doc = nlp(text)

        for token in doc:
            if token.pos_ == "ADJ":
                lemma = token.lemma_.lower()
                if lemma in synonym_to_trait:
                    traits_detected.add(synonym_to_trait[lemma])

        return list(traits_detected)

    def extract_entities(self, text: str) -> dict[str, list[str]]:
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
        traits = self.extract_traits(text)
        entities = self.extract_entities(text)

        # Update traits
        self.traits.extend(traits)
        self.traits = list(set(self.traits))

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

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of current persona traits and associations."""
        return {
            "traits": self.traits,
            "associations": {
                name: {"type": entity.entity_type, "traits": entity.traits}
                for name, entity in self.associations.items()
            },
        }

    def save_to_db(
        self,
        saver: BaseCheckpointSaver[V],
        config: RunnableConfig,
        metadata: Optional[dict[str, Any]] = None,
        new_versions: Optional[dict[str, V]] = None,
    ) -> None:
        """Save the current memory state to the codebase database using a checkpoint saver.

        Args:
            saver: An instance of SqliteSaver, PostgresSaver, etc.
            config: The checkpoint config (must include thread_id, etc.).
            metadata: Optional metadata dict.
            new_versions: Optional channel versions dict.
        """
        # Prepare the checkpoint data
        checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),  # or use the codebase's uuid6
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {
                "persona_traits": self.traits,
                "persona_associations": {
                    k: v.model_dump() for k, v in self.associations.items()
                },
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        if metadata is None:
            metadata = {}
        if new_versions is None:
            new_versions = {}

        # Save to the database
        saver.put(config, checkpoint, metadata, new_versions)

    async def save_to_db_async(
        self,
        saver: BaseCheckpointSaver[V],
        config: RunnableConfig,
        metadata: Optional[dict[str, Any]] = None,
        new_versions: Optional[dict[str, V]] = None,
    ) -> None:
        """
        Asynchronously save the current memory state to the codebase database using a checkpoint saver.

        Args:
            saver: An instance of SqliteSaver, PostgresSaver, etc. that supports async operations.
            config: The checkpoint config (must include thread_id, etc.).
            metadata: Optional metadata dict.
            new_versions: Optional channel versions dict.
        """
        # Prepare the checkpoint data
        checkpoint = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {
                "persona_traits": self.traits,
                "persona_associations": {
                    k: v.model_dump() for k, v in self.associations.items()
                },
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        if metadata is None:
            metadata = {}
        if new_versions is None:
            new_versions = {}

        # Save to the database asynchronously
        await saver.aput(config, checkpoint, metadata, new_versions)

    @classmethod
    def load_from_db(
        cls, saver: BaseCheckpointSaver[V], config: RunnableConfig
    ) -> "PersonaMemory":
        """Load a memory state from the database.

        Args:
            saver: An instance of SqliteSaver, PostgresSaver, etc.
            config: The checkpoint config (must include thread_id, etc.).

        Returns:
            PersonaMemory: A new instance loaded from the database.
        """
        checkpoint = saver.get(config)
        if checkpoint is None:
            return cls()

        channel_values = checkpoint["channel_values"]
        traits = channel_values.get("persona_traits", [])
        associations = {
            k: AssociatedEntity(**v)
            for k, v in channel_values.get("persona_associations", {}).items()
        }

        return cls(traits=traits, associations=associations)

    @classmethod
    async def load_from_db_async(
        cls, saver: BaseCheckpointSaver[V], config: RunnableConfig
    ) -> "PersonaMemory":
        """
        Asynchronously load a memory state from the database.

        Args:
            saver: An instance of SqliteSaver, PostgresSaver, etc. that supports async operations.
            config: The checkpoint config (must include thread_id, etc.).

        Returns:
            PersonaMemory: A new instance loaded from the database.
        """
        checkpoint = await saver.aget(config)
        if checkpoint is None:
            return cls()

        channel_values = checkpoint["channel_values"]
        traits = channel_values.get("persona_traits", [])
        associations = {
            k: AssociatedEntity(**v)
            for k, v in channel_values.get("persona_associations", {}).items()
        }

        return cls(traits=traits, associations=associations)
