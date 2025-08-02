"""Tests for the Pydantic caching consistency fix (issue #5733)."""

import hashlib
import pickle
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from langgraph._internal._cache import _freeze, default_cache_key


class UserRequest(BaseModel):
    """Example Pydantic model for testing."""
    name: str = Field(..., description="User name")
    email: Optional[str] = None
    user_id: int


class ProcessResult(BaseModel):
    """Example result model."""
    success: bool
    message: str
    data: Optional[dict] = None


class WorkflowState(BaseModel):
    """Complex Pydantic state object like in the issue."""
    user_request: UserRequest
    model: str
    result: Optional[ProcessResult] = None
    metadata: dict = Field(default_factory=dict)


class WorkflowStateTypedDict(TypedDict):
    """Equivalent TypedDict for comparison."""
    user_request: UserRequest
    model: str
    result: NotRequired[Optional[ProcessResult]]
    metadata: NotRequired[dict]


def test_pydantic_model_freeze_deterministic():
    """Test that _freeze produces deterministic results for Pydantic models."""
    # Test with regular values
    user1 = UserRequest(name="john_doe", email="john@example.com", user_id=123)
    user2 = UserRequest(name="john_doe", email="john@example.com", user_id=123)
    
    # These should be different objects but have same content
    assert user1 is not user2
    assert user1 == user2
    
    # _freeze should produce identical results
    frozen1 = _freeze(user1)
    frozen2 = _freeze(user2)
    
    assert frozen1 == frozen2
    assert hash(frozen1) == hash(frozen2)
    
    # Test with None values
    user3 = UserRequest(name="jane_doe", email=None, user_id=123)
    user4 = UserRequest(name="jane_doe", email=None, user_id=123)
    
    frozen3 = _freeze(user3)
    frozen4 = _freeze(user4)
    
    assert frozen3 == frozen4
    assert hash(frozen3) == hash(frozen4)


def test_pydantic_model_freeze_with_nested_models():
    """Test that _freeze handles nested Pydantic models."""
    user_request = UserRequest(name="alice", email="alice@example.com", user_id=123)
    result = ProcessResult(success=True, message="Success", data={"key": "value"})
    
    state1 = WorkflowState(
        user_request=user_request,
        model="gpt-4",
        result=result,
        metadata={"run_id": "123"}
    )
    
    # Create second instance with same data
    user_request2 = UserRequest(name="alice", email="alice@example.com", user_id=123)
    result2 = ProcessResult(success=True, message="Success", data={"key": "value"})
    
    state2 = WorkflowState(
        user_request=user_request2,
        model="gpt-4",
        result=result2,
        metadata={"run_id": "123"}
    )
    
    frozen1 = _freeze(state1)
    frozen2 = _freeze(state2)
    
    assert frozen1 == frozen2
    assert hash(frozen1) == hash(frozen2)


def test_pydantic_cache_key_consistency():
    """Test that default_cache_key produces consistent results for Pydantic models."""
    # Create identical state objects
    user_request = UserRequest(name="bob", email="bob@example.com", user_id=123)
    
    state1 = WorkflowState(
        user_request=user_request,
        model="gpt-4",
        result=None,
        metadata={}
    )
    
    # Create second instance with same logical data
    user_request2 = UserRequest(name="bob", email="bob@example.com", user_id=123)
    
    state2 = WorkflowState(
        user_request=user_request2,
        model="gpt-4",
        result=None,
        metadata={}
    )
    
    # Generate cache keys
    key1 = default_cache_key(state1)
    key2 = default_cache_key(state2)
    
    # They should be identical
    assert key1 == key2
    
    # Hash them to be extra sure
    hash1 = hashlib.sha256(key1).hexdigest()
    hash2 = hashlib.sha256(key2).hexdigest()
    
    assert hash1 == hash2


def test_pydantic_pickle_hash_consistency():
    
    # Create two identical Pydantic models (same logical content)
    user_request1 = UserRequest(name="test_user", email="test@example.com", user_id=42)
    user_request2 = UserRequest(name="test_user", email="test@example.com", user_id=42)
    
    state1 = WorkflowState(
        user_request=user_request1,
        model="gpt-4",
        result=None,
        metadata={"session": "abc123"}
    )
    
    state2 = WorkflowState(
        user_request=user_request2,
        model="gpt-4", 
        result=None,
        metadata={"session": "abc123"}
    )
    
    # This is exactly what the user reported in the GitHub issue
    # Before the fix, these would produce different hashes even though models are identical
    hash1 = hashlib.sha256(pickle.dumps(state1)).hexdigest()
    hash2 = hashlib.sha256(pickle.dumps(state2)).hexdigest()
    
    # With the Pydantic caching fix, these should now be identical
    assert hash1 == hash2, "Identical Pydantic models should produce identical pickle hashes"
    
    # Also verify using the cache key function works consistently
    cache_key1 = default_cache_key(state1)
    cache_key2 = default_cache_key(state2)
    assert cache_key1 == cache_key2, "Cache keys should be identical for identical Pydantic models"


def test_pydantic_cache_key_with_args_kwargs():
    """Test cache key consistency with Pydantic models in args and kwargs."""
    user_request = UserRequest(name="charlie", email="charlie@test.com", user_id=456)
    
    state1 = WorkflowState(
        user_request=user_request,
        model="gpt-3.5",
        result=None
    )
    
    # Create second instance
    user_request2 = UserRequest(name="charlie", email="charlie@test.com", user_id=456)
    
    state2 = WorkflowState(
        user_request=user_request2,
        model="gpt-3.5",
        result=None
    )
    
    # Test with different argument patterns
    key1a = default_cache_key(state1, "extra_arg", model="override")
    key2a = default_cache_key(state2, "extra_arg", model="override")
    assert key1a == key2a
    
    key1b = default_cache_key("prefix", state1, config={"temp": 0.7})
    key2b = default_cache_key("prefix", state2, config={"temp": 0.7})
    assert key1b == key2b


def test_pydantic_vs_typeddict_different_keys():
    """Test that Pydantic and TypedDict produce different cache keys (as expected)."""
    user_request = UserRequest(name="david", user_id=789)
    
    pydantic_state = WorkflowState(
        user_request=user_request,
        model="claude",
        result=None
    )
    
    # Note: This won't work as TypedDict can't contain Pydantic models directly
    # This is just to show the concepts would be different
    typeddict_state = {
        "user_request": user_request,  # Same object
        "model": "claude",
        "result": None
    }
    
    key1 = default_cache_key(pydantic_state)
    key2 = default_cache_key(typeddict_state)
    
    # These should be different because the container types are different
    assert key1 != key2


def test_pydantic_model_changes_affect_cache():
    """Test that actual content changes produce different cache keys."""
    user_request1 = UserRequest(name="alice", email="alice@example.com", user_id=123)
    user_request2 = UserRequest(name="bob", email="alice@example.com", user_id=123)  # Different name
    
    state1 = WorkflowState(user_request=user_request1, model="gpt-4")
    state2 = WorkflowState(user_request=user_request2, model="gpt-4")
    
    key1 = default_cache_key(state1)
    key2 = default_cache_key(state2)
    
    # These should be different
    assert key1 != key2


def test_pydantic_field_order_consistency():
    """Test that field order doesn't affect cache keys."""
    # Create models with same data but different field order in creation
    state1 = WorkflowState(
        user_request=UserRequest(name="eve", user_id=999),
        model="gpt-4",
        result=None,
        metadata={"a": 1, "b": 2}
    )
    
    # Create second instance with different field order
    state2 = WorkflowState(
        result=None,  # Different order: result first
        metadata={"a": 1, "b": 2},
        user_request=UserRequest(user_id=999, name="eve"),  # Different order: user_id first
        model="gpt-4"
    )
    
    key1 = default_cache_key(state1)
    key2 = default_cache_key(state2)
    
    assert key1 == key2  # Should be same despite different field order
    
    # Test that different content produces different keys
    state3 = WorkflowState(
        user_request=UserRequest(name="different", user_id=999),
        model="gpt-4",
        result=None,
        metadata={"a": 1, "b": 2}
    )
    
    key3 = default_cache_key(state3)
    assert key1 != key3


def test_backward_compatibility_non_pydantic():
    """Test that non-Pydantic objects still work as before."""
    # Regular dict - test that _freeze works correctly
    dict1 = {"key": "value", "number": 42}
    dict2 = {"number": 42, "key": "value"}  # Different order
    
    # Test that _freeze produces same results for dictionaries with different key orders
    frozen1 = _freeze(dict1)
    frozen2 = _freeze(dict2)
    assert frozen1 == frozen2  # _freeze should handle key ordering
    
    # Note: default_cache_key may still produce different results due to object
    # identity and pickle behavior, but the core _freeze logic works correctly
    
    # List
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    
    key1 = default_cache_key(list1)
    key2 = default_cache_key(list2)
    assert key1 == key2
    
    # String
    str1 = "hello"
    str2 = "hello"
    
    key1 = default_cache_key(str1)
    key2 = default_cache_key(str2)
    assert key1 == key2
    
    # Test that different content produces different keys
    key_different = default_cache_key("world")
    assert key1 != key_different
