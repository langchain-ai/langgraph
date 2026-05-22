from typing import Any

import orjson
import pytest
from pydantic import BaseModel

from langgraph_sdk.client import _aencode_json


async def _serde_roundtrip(data: Any):
    _, body = await _aencode_json(data)
    return orjson.loads(body)  # ty: ignore[invalid-argument-type]


async def test_serde_basic():
    # Test basic serialization
    data = {"key": "value", "number": 42}
    assert await _serde_roundtrip(data) == data


async def test_serde_pydantic():
    # Test serialization with Pydantic model (if available)

    class TestModel(BaseModel):
        name: str
        age: int

    model = TestModel(name="test", age=25)
    result = await _serde_roundtrip(model)
    assert result["name"] == "test"
    assert result["age"] == 25

    nested_result = await _serde_roundtrip({"data": model})
    assert nested_result["data"]["name"] == "test"
    assert nested_result["data"]["age"] == 25


async def test_serde_dataclass():
    from dataclasses import dataclass

    @dataclass
    class TestDataClass:
        name: str
        age: int

    data = TestDataClass(name="test", age=25)
    result = await _serde_roundtrip(data)
    assert result["name"] == "test"
    assert result["age"] == 25

    nested_result = await _serde_roundtrip({"data": data})
    assert nested_result["data"]["name"] == "test"
    assert nested_result["data"]["age"] == 25


async def test_serde_pydantic_cls_fails():
    # Test that serialization fails gracefully for Pydantic model when not available
    class TestModel(BaseModel):
        name: str

    with pytest.raises(TypeError, match="Type is not JSON serializable"):
        await _serde_roundtrip({"foo": TestModel})
