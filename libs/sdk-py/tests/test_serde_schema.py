import concurrent.futures
from dataclasses import dataclass
from typing import runtime_checkable

import mypy.api
from pydantic import BaseModel

from langgraph_sdk.schema import (
    _BaseModelLike,
    _DataclassLike,
)


def rc(cls: type) -> type:
    return runtime_checkable(cls)


class MyModel(BaseModel):
    foo: str


def test_base_model_like():
    assert isinstance(MyModel(foo="test"), rc(_BaseModelLike))


@dataclass
class MyDataclass:
    foo: str


def test_dataclass_like():
    assert isinstance(MyDataclass(foo="test"), rc(_DataclassLike))


def test_mypy_type():
    pydantic_example = """
from langgraph_sdk import get_client
from pydantic import BaseModel

class Foo(BaseModel):
    bar: str

# This should type-check without errors
client = get_client()

client.runs.stream(None, "agent", input=Foo(bar="test"))"""

    typed_dict_example = """
from langgraph_sdk import get_client
from typing import TypedDict

class MyTypedDict(TypedDict):
    key: str

# This should type-check without errors
client = get_client()
client.runs.stream(None, "agent", input=MyTypedDict(key="value"))
    """

    dataclass_example = """
from langgraph_sdk import get_client
from dataclasses import dataclass

@dataclass
class MyDataclass:
    key: str

# This should type-check without errors
client = get_client()
client.runs.stream(None, "agent", input=MyDataclass(key="value"))
    """

    unsupported_primitive = """
from langgraph_sdk import get_client

# This should type-check without errors
client = get_client()
client.runs.stream(None, "agent", input="any string")
    """

    unsupported_py_object = """
from langgraph_sdk import get_client

class MyObject:
    def __init__(self, value: str):
        self.value = value

client = get_client()
client.runs.stream(None, "agent", input=MyObject("test"))
"""

    def check(s: str, expect_fail: bool = False):
        result = mypy.api.run(["-c", s, "--strict"])
        if expect_fail:
            assert result[2] != 0, (
                f"Expected mypy to find errors but it passed: {result[0]} {result[1]} (exit code: {result[2]})"
            )
        else:
            assert result[2] == 0, (
                f"Mypy found errors: {result[0]} {result[1]} (exit code: {result[2]})"
            )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(check, pydantic_example))
        futures.append(executor.submit(check, typed_dict_example))
        futures.append(executor.submit(check, dataclass_example))
        futures.append(executor.submit(check, unsupported_primitive, expect_fail=True))
        futures.append(executor.submit(check, unsupported_py_object, expect_fail=True))

        for future in concurrent.futures.as_completed(futures):
            future.result()
