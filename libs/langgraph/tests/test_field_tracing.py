"""Tests for field-level tracing annotations."""

from typing import Annotated

import pytest
from typing_extensions import TypedDict

from langgraph.tracing import (
    AsAttachment,
    NotTraced,
    RootOnly,
    Traced,
    TraceLevel,
    filter_state_for_tracing,
    get_trace_annotation,
)


class TestTraceAnnotations:
    """Test tracing annotation classes."""

    def test_traced_annotation(self):
        """Test Traced annotation."""
        traced = Traced()
        assert traced.level == TraceLevel.TRACED
        assert traced.should_trace(is_root=True)
        assert traced.should_trace(is_root=False)
        assert traced.get_trace_value("test", is_root=True) == "test"
        assert traced.get_trace_value("test", is_root=False) == "test"

    def test_not_traced_annotation(self):
        """Test NotTraced annotation."""
        not_traced = NotTraced()
        assert not_traced.level == TraceLevel.NOT_TRACED
        assert not not_traced.should_trace(is_root=True)
        assert not not_traced.should_trace(is_root=False)
        assert not_traced.get_trace_value("test", is_root=True) == "[NOT TRACED]"
        assert not_traced.get_trace_value("test", is_root=False) == "[NOT TRACED]"

    def test_not_traced_custom_mask(self):
        """Test NotTraced with custom mask value."""
        not_traced = NotTraced(mask_value="<hidden>")
        assert not_traced.get_trace_value("test", is_root=True) == "<hidden>"
        assert not_traced.get_trace_value("test", is_root=False) == "<hidden>"

    def test_root_only_annotation(self):
        """Test RootOnly annotation."""
        root_only = RootOnly()
        assert root_only.level == TraceLevel.ROOT_ONLY
        assert root_only.should_trace(is_root=True)
        assert not root_only.should_trace(is_root=False)
        assert root_only.get_trace_value("test", is_root=True) == "test"
        assert (
            root_only.get_trace_value("test", is_root=False) == "[MASKED IN CHILD RUN]"
        )

    def test_as_attachment_annotation(self):
        """Test AsAttachment annotation."""
        as_attachment = AsAttachment()
        assert as_attachment.level == TraceLevel.AS_ATTACHMENT
        assert as_attachment.should_trace(is_root=True)
        assert not as_attachment.should_trace(is_root=False)
        assert as_attachment.get_trace_value("test", is_root=True) == "test"
        assert (
            as_attachment.get_trace_value("test", is_root=False) == "[SEE ROOT TRACE]"
        )

    def test_as_attachment_custom_mask(self):
        """Test AsAttachment with custom mask value."""
        as_attachment = AsAttachment(mask_value="<see parent>")
        assert as_attachment.get_trace_value("test", is_root=True) == "test"
        assert as_attachment.get_trace_value("test", is_root=False) == "<see parent>"


class TestGetTraceAnnotation:
    """Test get_trace_annotation helper function."""

    def test_no_annotation(self):
        """Test type without annotation."""
        assert get_trace_annotation(str) is None
        assert get_trace_annotation(int) is None

    def test_with_traced_annotation(self):
        """Test Annotated type with Traced."""
        ann_type = Annotated[str, Traced()]
        annotation = get_trace_annotation(ann_type)
        assert annotation is not None
        assert annotation.level == TraceLevel.TRACED

    def test_with_not_traced_annotation(self):
        """Test Annotated type with NotTraced."""
        ann_type = Annotated[bytes, NotTraced()]
        annotation = get_trace_annotation(ann_type)
        assert annotation is not None
        assert annotation.level == TraceLevel.NOT_TRACED

    def test_with_root_only_annotation(self):
        """Test Annotated type with RootOnly."""
        ann_type = Annotated[str, RootOnly()]
        annotation = get_trace_annotation(ann_type)
        assert annotation is not None
        assert annotation.level == TraceLevel.ROOT_ONLY

    def test_with_as_attachment_annotation(self):
        """Test Annotated type with AsAttachment."""
        ann_type = Annotated[dict, AsAttachment()]
        annotation = get_trace_annotation(ann_type)
        assert annotation is not None
        assert annotation.level == TraceLevel.AS_ATTACHMENT

    def test_with_multiple_metadata(self):
        """Test Annotated type with multiple metadata items."""
        # The trace annotation should be found even with other metadata
        ann_type = Annotated[str, "some doc", NotTraced(), "other metadata"]
        annotation = get_trace_annotation(ann_type)
        assert annotation is not None
        assert annotation.level == TraceLevel.NOT_TRACED


class TestFilterStateForTracing:
    """Test filter_state_for_tracing function."""

    def test_non_dict_state(self):
        """Test that non-dict state is returned as-is."""
        assert filter_state_for_tracing("string", {}, is_root=True) == "string"
        assert filter_state_for_tracing(42, {}, is_root=False) == 42
        assert filter_state_for_tracing(None, {}, is_root=True) is None

    def test_dict_without_type_hints(self):
        """Test dict without type hints is returned as-is."""
        state = {"key": "value", "num": 42}
        result = filter_state_for_tracing(state, {}, is_root=True)
        assert result == state

    def test_filter_not_traced_field(self):
        """Test filtering NotTraced fields."""
        type_hints = {
            "query": str,
            "pdf": Annotated[bytes, NotTraced()],
        }
        state = {"query": "test query", "pdf": b"large pdf content"}

        # Root run - should still mask NotTraced
        result = filter_state_for_tracing(state, type_hints, is_root=True)
        assert result["query"] == "test query"
        assert result["pdf"] == "[NOT TRACED]"

        # Child run - should also mask NotTraced
        result = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result["query"] == "test query"
        assert result["pdf"] == "[NOT TRACED]"

    def test_filter_root_only_field(self):
        """Test filtering RootOnly fields."""
        type_hints = {
            "query": str,
            "document": Annotated[str, RootOnly()],
        }
        state = {"query": "test query", "document": "long document"}

        # Root run - should show document
        result = filter_state_for_tracing(state, type_hints, is_root=True)
        assert result["query"] == "test query"
        assert result["document"] == "long document"

        # Child run - should mask document
        result = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result["query"] == "test query"
        assert result["document"] == "[MASKED IN CHILD RUN]"

    def test_filter_as_attachment_field(self):
        """Test filtering AsAttachment fields."""
        type_hints = {
            "query": str,
            "dataframe": Annotated[dict, AsAttachment(mask_value="<dataframe>")],
        }
        state = {
            "query": "test query",
            "dataframe": {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        }

        # Root run - should show dataframe
        result = filter_state_for_tracing(state, type_hints, is_root=True)
        assert result["query"] == "test query"
        assert result["dataframe"] == {"col1": [1, 2, 3], "col2": [4, 5, 6]}

        # Child run - should mask dataframe
        result = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result["query"] == "test query"
        assert result["dataframe"] == "<dataframe>"

    def test_mixed_annotations(self):
        """Test state with mixed annotation types."""
        type_hints = {
            "query": str,
            "pdf": Annotated[bytes, NotTraced()],
            "context": Annotated[str, RootOnly()],
            "results": Annotated[list, AsAttachment()],
            "score": float,
        }
        state = {
            "query": "find info",
            "pdf": b"pdf bytes",
            "context": "background context",
            "results": [{"doc": 1}, {"doc": 2}],
            "score": 0.95,
        }

        # Root run
        result_root = filter_state_for_tracing(state, type_hints, is_root=True)
        assert result_root["query"] == "find info"
        assert result_root["pdf"] == "[NOT TRACED]"
        assert result_root["context"] == "background context"
        assert result_root["results"] == [{"doc": 1}, {"doc": 2}]
        assert result_root["score"] == 0.95

        # Child run
        result_child = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result_child["query"] == "find info"
        assert result_child["pdf"] == "[NOT TRACED]"
        assert result_child["context"] == "[MASKED IN CHILD RUN]"
        assert result_child["results"] == "[SEE ROOT TRACE]"
        assert result_child["score"] == 0.95

    def test_field_not_in_type_hints(self):
        """Test that fields not in type hints are passed through."""
        type_hints = {
            "known_field": Annotated[str, NotTraced()],
        }
        state = {
            "known_field": "value",
            "unknown_field": "should pass through",
        }

        result = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result["known_field"] == "[NOT TRACED]"
        assert result["unknown_field"] == "should pass through"


class TestTypedDictIntegration:
    """Test integration with TypedDict state schemas."""

    def test_typed_dict_with_annotations(self):
        """Test that TypedDict with annotations works correctly."""

        class AgentState(TypedDict):
            query: str
            pdf_content: Annotated[bytes, NotTraced()]
            intermediate_results: Annotated[list, RootOnly()]
            final_answer: str

        # Simulate getting type hints from TypedDict
        from typing import get_type_hints

        type_hints = get_type_hints(AgentState, include_extras=True)

        state = {
            "query": "What is in the PDF?",
            "pdf_content": b"large pdf bytes" * 1000,
            "intermediate_results": ["step1", "step2", "step3"],
            "final_answer": "The answer is...",
        }

        # Root run - intermediate_results visible, pdf_content masked
        result_root = filter_state_for_tracing(state, type_hints, is_root=True)
        assert result_root["query"] == "What is in the PDF?"
        assert result_root["pdf_content"] == "[NOT TRACED]"
        assert result_root["intermediate_results"] == ["step1", "step2", "step3"]
        assert result_root["final_answer"] == "The answer is..."

        # Child run - intermediate_results masked, pdf_content masked
        result_child = filter_state_for_tracing(state, type_hints, is_root=False)
        assert result_child["query"] == "What is in the PDF?"
        assert result_child["pdf_content"] == "[NOT TRACED]"
        assert result_child["intermediate_results"] == "[MASKED IN CHILD RUN]"
        assert result_child["final_answer"] == "The answer is..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
