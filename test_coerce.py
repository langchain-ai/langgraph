# Test file to verify the _coerce_context function works correctly
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
from typing_extensions import TypedDict, is_typeddict
from dataclasses import is_dataclass
from inspect import isclass


def _coerce_context(
    context_schema: type | None, context: Any
) -> Any | None:
    """Coerce context input to the appropriate schema type.
    
    Args:
        context_schema: The schema type to coerce to (BaseModel, dataclass, or TypedDict)
        context: The context value to coerce
        
    Returns:
        The coerced context value or None if context is None
    """
    if context is None or context_schema is None:
        return context
    
    # If context is a dict and we have a schema, try to coerce
    if isinstance(context, dict):
        # Handle TypedDict first - pass through as-is since TypedDicts are just dicts at runtime
        if is_typeddict(context_schema):
            return context
        # Handle Pydantic BaseModel
        elif isclass(context_schema) and issubclass(context_schema, BaseModel):
            return context_schema(**context)
        # Handle dataclass
        elif is_dataclass(context_schema):
            return context_schema(**context)
    
    # If context is already the correct type, return as-is
    # This check is done after dict check to avoid TypedDict isinstance issues
    try:
        if isinstance(context, context_schema):
            return context
    except TypeError:
        # TypedDict raises TypeError on isinstance check
        pass
    
    # If we can't coerce, return the context as-is
    return context


# Test with dataclass
@dataclass
class DataclassContext:
    api_key: str
    user_id: str


# Test with Pydantic model
class PydanticContext(BaseModel):
    api_key: str
    user_id: str


# Test with TypedDict
class TypedDictContext(TypedDict):
    api_key: str
    user_id: str


# Test the function
if __name__ == "__main__":
    test_dict = {"api_key": "test_key", "user_id": "test_user"}
    
    # Test dataclass coercion
    dc_result = _coerce_context(DataclassContext, test_dict)
    print(f"Dataclass result: {dc_result}")
    print(f"Is DataclassContext instance: {isinstance(dc_result, DataclassContext)}")
    
    # Test Pydantic coercion
    pyd_result = _coerce_context(PydanticContext, test_dict)
    print(f"Pydantic result: {pyd_result}")
    print(f"Is PydanticContext instance: {isinstance(pyd_result, PydanticContext)}")
    
    # Test TypedDict coercion (should return dict as-is)
    td_result = _coerce_context(TypedDictContext, test_dict)
    print(f"TypedDict result: {td_result}")
    print(f"Is dict: {isinstance(td_result, dict)}")
    
    # Test None context
    none_result = _coerce_context(DataclassContext, None)
    print(f"None context result: {none_result}")
    
    # Test already correct type
    dc_instance = DataclassContext(api_key="key", user_id="user")
    same_result = _coerce_context(DataclassContext, dc_instance)
    print(f"Already correct type: {same_result is dc_instance}")

