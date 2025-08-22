import json
import re
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel


def _camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _convert_keys(obj: Any) -> Any:
    """Recursively convert dict keys to snake_case."""
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys(i) for i in obj]
    else:
        return obj


def load_spec(spec_name: str, as_model: Type[BaseModel]) -> list[BaseModel]:
    with (Path(__file__).parent / "specifications" / f"{spec_name}.json").open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
        converted = _convert_keys(data)
        print("CONVERTED: ", converted)
        return [as_model(**item) for item in converted]
