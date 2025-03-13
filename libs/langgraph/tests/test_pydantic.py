import sys
import typing

import pydantic
import typing_extensions

from langgraph.utils.pydantic import is_supported_by_pydantic


def test_is_supported_by_pydantic() -> None:
    """Test if types are supported by pydantic."""

    class TypedDictExtensions(typing_extensions.TypedDict):
        x: int

    assert is_supported_by_pydantic(TypedDictExtensions) is True

    class VanillaClass:
        x: int

    assert is_supported_by_pydantic(VanillaClass) is False

    class BuiltinTypedDict(typing.TypedDict):  # noqa: TID251
        x: int

    if sys.version_info >= (3, 12):
        assert is_supported_by_pydantic(BuiltinTypedDict) is True
    else:
        assert is_supported_by_pydantic(BuiltinTypedDict) is False

    class PydanticModel(pydantic.BaseModel):
        x: int

    assert is_supported_by_pydantic(PydanticModel) is True

    if hasattr(pydantic, "v1"):

        class PydanticModelV1(pydantic.v1.BaseModel):
            x: int

        assert is_supported_by_pydantic(PydanticModelV1) is False

    assert is_supported_by_pydantic(int) is False
