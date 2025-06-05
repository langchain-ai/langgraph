from __future__ import annotations

import sys
import typing
import warnings
from contextlib import nullcontext
from dataclasses import is_dataclass
from functools import lru_cache
from typing import (
    Any,
    cast,
    overload,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)
from pydantic import (
    create_model as _create_model_base,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)
from typing_extensions import TypedDict


@overload
def get_fields(model: type[BaseModel]) -> dict[str, FieldInfo]: ...


@overload
def get_fields(model: BaseModel) -> dict[str, FieldInfo]: ...


def get_fields(
    model: type[BaseModel] | BaseModel,
) -> dict[str, FieldInfo]:
    """Get the field names of a Pydantic model."""
    if hasattr(model, "model_fields"):
        return model.model_fields

    if hasattr(model, "__fields__"):
        return model.__fields__
    msg = f"Expected a Pydantic model. Got {type(model)}"
    raise TypeError(msg)


_SchemaConfig = ConfigDict(
    arbitrary_types_allowed=True, frozen=True, protected_namespaces=()
)

NO_DEFAULT = object()


def _create_root_model(
    name: str,
    type_: Any,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    """Create a base class."""

    def schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
    ) -> dict[str, Any]:
        # Complains about schema not being defined in superclass
        schema_ = super(cls, cls).schema(  # type: ignore[misc]
            by_alias=by_alias, ref_template=ref_template
        )
        schema_["title"] = name
        return schema_

    def model_json_schema(
        cls: type[BaseModel],
        by_alias: bool = True,  # noqa: FBT001,FBT002
        ref_template: str = DEFAULT_REF_TEMPLATE,
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: JsonSchemaMode = "validation",
    ) -> dict[str, Any]:
        # Complains about model_json_schema not being defined in superclass
        schema_ = super(cls, cls).model_json_schema(  # type: ignore[misc]
            by_alias=by_alias,
            ref_template=ref_template,
            schema_generator=schema_generator,
            mode=mode,
        )
        schema_["title"] = name
        return schema_

    base_class_attributes = {
        "__annotations__": {"root": type_},
        "model_config": ConfigDict(arbitrary_types_allowed=True),
        "schema": classmethod(schema),
        "model_json_schema": classmethod(model_json_schema),
        "__module__": module_name or "langchain_core.runnables.utils",
    }

    if default_ is not NO_DEFAULT:
        base_class_attributes["root"] = default_
    with warnings.catch_warnings():
        custom_root_type = type(name, (RootModel,), base_class_attributes)
    return cast("type[BaseModel]", custom_root_type)


@lru_cache(maxsize=256)
def _create_root_model_cached(
    model_name: str,
    type_: Any,
    *,
    module_name: str | None = None,
    default_: object = NO_DEFAULT,
) -> type[BaseModel]:
    return _create_root_model(
        model_name, type_, default_=default_, module_name=module_name
    )


@lru_cache(maxsize=256)
def _create_model_cached(
    model_name: str,
    /,
    **field_definitions: Any,
) -> type[BaseModel]:
    return _create_model_base(
        model_name,
        __config__=_SchemaConfig,
        **_remap_field_definitions(field_definitions),
    )


# Reserved names should capture all the `public` names / methods that are
# used by BaseModel internally. This will keep the reserved names up-to-date.
# For reference, the reserved names are:
# "construct", "copy", "dict", "from_orm", "json", "parse_file", "parse_obj",
# "parse_raw", "schema", "schema_json", "update_forward_refs", "validate",
# "model_computed_fields", "model_config", "model_construct", "model_copy",
# "model_dump", "model_dump_json", "model_extra", "model_fields",
# "model_fields_set", "model_json_schema", "model_parametrized_name",
# "model_post_init", "model_rebuild", "model_validate", "model_validate_json",
# "model_validate_strings"
_RESERVED_NAMES = {key for key in dir(BaseModel) if not key.startswith("_")}


def _remap_field_definitions(field_definitions: dict[str, Any]) -> dict[str, Any]:
    """This remaps fields to avoid colliding with internal pydantic fields."""

    remapped = {}
    for key, value in field_definitions.items():
        if key.startswith("_") or key in _RESERVED_NAMES:
            # Let's add a prefix to avoid colliding with internal pydantic fields
            if isinstance(value, FieldInfo):
                msg = (
                    f"Remapping for fields starting with '_' or fields with a name "
                    f"matching a reserved name {_RESERVED_NAMES} is not supported if "
                    f" the field is a pydantic Field instance. Got {key}."
                )
                raise NotImplementedError(msg)
            type_, default_ = value
            remapped[f"private_{key}"] = (
                type_,
                Field(
                    default=default_,
                    alias=key,
                    serialization_alias=key,
                    title=key.lstrip("_").replace("_", " ").title(),
                ),
            )
        else:
            remapped[key] = value
    return remapped


def create_model(
    model_name: str,
    *,
    field_definitions: dict[str, Any] | None = None,
    root: Any | None = None,
) -> type[BaseModel]:
    """Create a pydantic model with the given field definitions.

    Attention:
        Please do not use outside of langchain packages. This API
        is subject to change at any time.

    Args:
        model_name: The name of the model.
        module_name: The name of the module where the model is defined.
            This is used by Pydantic to resolve any forward references.
        field_definitions: The field definitions for the model.
        root: Type for a root model (RootModel)

    Returns:
        Type[BaseModel]: The created model.
    """
    field_definitions = field_definitions or {}

    if root:
        if field_definitions:
            msg = (
                "When specifying __root__ no other "
                f"fields should be provided. Got {field_definitions}"
            )
            raise NotImplementedError(msg)

        if isinstance(root, tuple):
            kwargs = {"type_": root[0], "default_": root[1]}
        else:
            kwargs = {"type_": root}

        try:
            named_root_model = _create_root_model_cached(model_name, **kwargs)
        except TypeError:
            # something in the arguments into _create_root_model_cached is not hashable
            named_root_model = _create_root_model(
                model_name,
                **kwargs,
            )
        return named_root_model

    # No root, just field definitions
    names = set(field_definitions.keys())

    capture_warnings = False

    for name in names:
        # Also if any non-reserved name is used (e.g., model_id or model_name)
        if name.startswith("model"):
            capture_warnings = True

    with warnings.catch_warnings() if capture_warnings else nullcontext():
        if capture_warnings:
            warnings.filterwarnings(action="ignore")
        try:
            return _create_model_cached(model_name, **field_definitions)
        except TypeError:
            # something in field definitions is not hashable
            return _create_model_base(
                model_name,
                __config__=_SchemaConfig,
                **_remap_field_definitions(field_definitions),
            )


def is_supported_by_pydantic(type_: Any) -> bool:
    """Check if a given "complex" type is supported by pydantic.

    This will return False for primitive types like int, str, etc.

    The check is meant for container types like dataclasses, TypedDicts, etc.
    """
    if is_dataclass(type_):
        return True

    if isinstance(type_, type) and issubclass(type_, BaseModel):
        return True

    if hasattr(type_, "__orig_bases__"):
        for base in type_.__orig_bases__:
            if base is TypedDict:
                return True
            elif base is typing.TypedDict:  # noqa: TID251
                # ignoring TID251 since it's OK to use typing.TypedDict in this case.
                # Pydantic supports typing.TypedDict from Python 3.12
                # For older versions, only typing_extensions.TypedDict is supported.
                if sys.version_info >= (3, 12):
                    return True
    return False
