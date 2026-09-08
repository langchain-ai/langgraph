from enum import Enum, StrEnum

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class RegularEnum(Enum):
    PENDING = "pending"
    COMPLETED = "completed"


class StringEnum(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"


def test_strenum_serialization() -> None:
    """Test that StrEnum is preserved during serialization/deserialization."""
    serde = JsonPlusSerializer()

    regular = RegularEnum.PENDING
    string = StringEnum.PENDING

    # Serialize and deserialize
    dumped_reg = serde.dumps_typed(regular)
    loaded_reg = serde.loads_typed(dumped_reg)

    dumped_str = serde.dumps_typed(string)
    loaded_str = serde.loads_typed(dumped_str)

    # RegularEnum should be preserved
    assert isinstance(loaded_reg, RegularEnum)
    assert loaded_reg == regular
    assert type(loaded_reg) == type(regular)

    # StrEnum should also be preserved
    assert isinstance(loaded_str, StringEnum), (
        f"Expected StringEnum, got {type(loaded_str)}"
    )
    assert loaded_str == string
    assert type(loaded_str) == type(string)


def test_strenum_in_dict() -> None:
    """Test StrEnum serialization when nested in a dict."""
    serde = JsonPlusSerializer()

    data = {
        "regular": RegularEnum.PENDING,
        "string": StringEnum.PENDING,
    }

    dumped = serde.dumps_typed(data)
    loaded = serde.loads_typed(dumped)

    assert isinstance(loaded["regular"], RegularEnum)
    assert isinstance(loaded["string"], StringEnum), (
        f"Expected StringEnum, got {type(loaded['string'])}"
    )

