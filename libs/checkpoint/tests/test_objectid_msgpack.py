"""Tests for bson.ObjectId msgpack serialization (issue #7467).

Skipped automatically when the bson package is not installed.
"""
import pytest

bson = pytest.importorskip("bson", reason="bson package not installed")


class TestObjectIdMsgpackRoundTrip:
    """bson.ObjectId must survive a JsonPlusSerializer dumps/loads round-trip."""

    def test_objectid_roundtrip_standalone(self) -> None:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        obj_id = bson.ObjectId()
        serde = JsonPlusSerializer()
        kind, data = serde.dumps(obj_id)
        result = serde.loads(kind, data)
        assert isinstance(result, bson.ObjectId)
        assert result == obj_id

    def test_objectid_inside_pydantic_model(self) -> None:
        from typing import Annotated, Any

        from pydantic import BaseModel
        from pydantic.json_schema import JsonSchemaValue
        from pydantic_core import core_schema

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        class _ObjectIdAnnotation:
            @classmethod
            def validate(cls, v: Any, handler):
                if isinstance(v, bson.ObjectId):
                    return v
                s = handler(v)
                if bson.ObjectId.is_valid(s):
                    return bson.ObjectId(s)
                raise ValueError("Invalid ObjectId")

            @classmethod
            def __get_pydantic_core_schema__(cls, source_type, _handler):
                return core_schema.no_info_wrap_validator_function(
                    cls.validate,
                    core_schema.str_schema(),
                    serialization=core_schema.to_string_ser_schema(),
                )

            @classmethod
            def __get_pydantic_json_schema__(cls, _core_schema, handler) -> JsonSchemaValue:
                return handler(core_schema.str_schema())

        PydanticObjectId = Annotated[bson.ObjectId, _ObjectIdAnnotation]

        class ModelWithObjectId(BaseModel):
            object_id: PydanticObjectId

        original = ModelWithObjectId(object_id=bson.ObjectId())
        serde = JsonPlusSerializer()
        kind, data = serde.dumps(original)
        result = serde.loads(kind, data)
        assert isinstance(result, ModelWithObjectId)
        assert result.object_id == original.object_id

    def test_objectid_in_allowed_safe_types(self) -> None:
        from langgraph.checkpoint.serde._msgpack import SAFE_MSGPACK_TYPES

        assert ("bson.objectid", "ObjectId") in SAFE_MSGPACK_TYPES
