"""Unit tests for langgraph.prebuilt.responses module."""

import pytest
from pydantic import BaseModel

from langgraph.prebuilt.responses import (
    OutputToolBinding,
    ResponseFormat,
    SchemaSpec,
    ToolOutput,
)


class TestModel(BaseModel):
    """A test model for structured output."""

    name: str
    age: int
    email: str = "default@example.com"


class CustomModel(BaseModel):
    """Custom model with a custom docstring."""

    value: float
    description: str


class EmptyDocModel(BaseModel):
    # No custom docstring, should have no description in tool
    data: str


class TestSchemaSpec:
    """Test SchemaSpec dataclass."""

    def test_basic_creation(self):
        """Test basic SchemaSpec creation."""
        schema = SchemaSpec(schema=TestModel)
        assert schema.schema == TestModel
        assert schema.name is None
        assert schema.description is None
        assert schema.strict is False

    def test_creation_with_all_fields(self):
        """Test SchemaSpec creation with all fields."""
        schema = SchemaSpec(
            schema=TestModel,
            name="custom_test_model",
            description="A custom description",
            strict=True,
        )
        assert schema.schema == TestModel
        assert schema.name == "custom_test_model"
        assert schema.description == "A custom description"
        assert schema.strict is True


class TestUsingToolStrategy:
    """Test UsingToolStrategy dataclass."""

    def test_basic_creation(self):
        """Test basic UsingToolStrategy creation."""
        schema = SchemaSpec(schema=TestModel)
        strategy = ToolOutput(schemas=[schema])
        assert len(strategy.schemas) == 1
        assert strategy.schemas[0] == schema
        assert strategy.tool_choice == "required"  # default

    def test_creation_with_auto_tool_choice(self):
        """Test UsingToolStrategy creation with auto tool choice."""
        schema = SchemaSpec(schema=TestModel)
        strategy = ToolOutput(schemas=[schema], tool_choice="auto")
        assert strategy.tool_choice == "auto"

    def test_multiple_schemas(self):
        """Test UsingToolStrategy with multiple schemas."""
        schema1 = SchemaSpec(schema=TestModel)
        schema2 = SchemaSpec(schema=CustomModel)
        strategy = ToolOutput(schemas=[schema1, schema2])
        assert len(strategy.schemas) == 2


class TestOutputToolBinding:
    """Test OutputToolBinding dataclass and its methods."""

    def test_from_schema_spec_basic(self):
        """Test basic OutputToolBinding creation from SchemaSpec."""
        schema_spec = SchemaSpec(schema=TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.schema == TestModel
        assert tool_binding.schema_kind == "pydantic"
        assert tool_binding.tool is not None
        assert tool_binding.tool.name == "TestModel"

    def test_from_schema_spec_with_custom_name(self):
        """Test OutputToolBinding creation with custom name."""
        schema_spec = SchemaSpec(schema=TestModel, name="custom_tool_name")
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)
        assert tool_binding.tool.name == "custom_tool_name"

    def test_from_schema_spec_with_custom_description(self):
        """Test OutputToolBinding creation with custom description."""
        schema_spec = SchemaSpec(
            schema=TestModel, description="Custom tool description"
        )
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.tool.description == "Custom tool description"

    def test_from_schema_spec_with_model_docstring(self):
        """Test OutputToolBinding creation using model docstring as description."""
        schema_spec = SchemaSpec(schema=CustomModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        assert tool_binding.tool.description == "Custom model with a custom docstring."

    @pytest.mark.skip(
        reason="Need to fix bug in langchain-core for inheritance of doc-strings."
    )
    def test_from_schema_spec_empty_docstring(self):
        """Test OutputToolBinding creation with model that has default docstring."""

        # Create a model with the same docstring as BaseModel
        class DefaultDocModel(BaseModel):
            # This should have the same docstring as BaseModel
            pass

        schema_spec = SchemaSpec(schema=DefaultDocModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        # Should use empty description when model has default BaseModel docstring
        assert tool_binding.tool.description == ""

    def test_parse_payload_pydantic_success(self):
        """Test successful parsing for Pydantic model."""
        schema_spec = SchemaSpec(schema=TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        tool_args = {"name": "John", "age": 30}
        result = tool_binding.parse_payload(tool_args)

        assert isinstance(result, TestModel)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "default@example.com"  # default value

    def test_parse_payload_pydantic_validation_error(self):
        """Test parsing failure for invalid Pydantic data."""
        schema_spec = SchemaSpec(schema=TestModel)
        tool_binding = OutputToolBinding.from_schema_spec(schema_spec)

        # Missing required field 'name'
        tool_args = {"age": 30}

        with pytest.raises(ValueError, match="Failed to parse tool args to TestModel"):
            tool_binding.parse_payload(tool_args)

    def test_parse_payload_invalid_kind(self):
        """Test parsing with invalid kind."""
        from unittest.mock import Mock

        mock_tool = Mock()

        tool_binding = OutputToolBinding(
            schema=TestModel,
            schema_kind="invalid_kind",  # type: ignore
            tool=mock_tool,
        )

        with pytest.raises(ValueError, match="Unsupported schema kind: invalid_kind"):
            tool_binding.parse_payload({"name": "test", "age": 25})

    def test_parse_payload_invalid_pydantic_schema(self):
        """Test parsing with invalid schema for pydantic kind."""
        from unittest.mock import Mock

        mock_tool = Mock()

        # Create tool binding with dict schema but pydantic kind
        tool_binding = OutputToolBinding(
            schema={"type": "object"}, schema_kind="pydantic", tool=mock_tool
        )

        with pytest.raises(
            ValueError, match="Expected Pydantic model class for 'pydantic' kind"
        ):
            tool_binding.parse_payload({"name": "test", "age": 25})


class TestResponseFormat:
    """Test ResponseFormat type alias."""

    def test_response_format_is_using_tool_strategy(self):
        """Test that ResponseFormat is aliased to UsingToolStrategy."""
        assert ResponseFormat is ToolOutput

    def test_can_create_response_format(self):
        """Test that we can create ResponseFormat instances."""
        schema = SchemaSpec(schema=TestModel)
        response_format = ResponseFormat(schemas=[schema])

        assert isinstance(response_format, ToolOutput)
        assert len(response_format.schemas) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_schemas_list(self) -> None:
        """Test UsingToolStrategy with empty schemas list."""
        strategy = ToolOutput([SchemaSpec(EmptyDocModel)])
        assert len(strategy.schemas) == 1

    @pytest.mark.skip(
        reason="Need to fix bug in langchain-core for inheritance of doc-strings."
    )
    def test_base_model_doc_constant(self) -> None:
        """Test that BASE_MODEL_DOC constant is set correctly."""
        binding = OutputToolBinding.from_schema_spec(SchemaSpec(EmptyDocModel))
        assert binding.tool.name == "EmptyDocModel"
        assert (
            binding.tool.description[:5] == ""
        )  # Should be empty for default docstring
