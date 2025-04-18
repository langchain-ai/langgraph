import functools
import logging
import weakref
from inspect import isclass
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    get_args,
    TypeVar,
    get_type_hints,
    Hashable,
    get_origin, Union,
)

from pydantic import BaseModel, Discriminator
from pydantic.fields import FieldInfo
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Annotated, Literal

__all__ = ["SchemaCoercionMapper"]

logger = logging.getLogger(__name__)

_cache: weakref.WeakKeyDictionary[Type[Any], dict[int, "SchemaCoercionMapper"]] = (
    weakref.WeakKeyDictionary()
)


class SchemaCoercionMapper:
    """Lightweight coercion of *dict* → *BaseModel* instances."""

    def __new__(
            cls,
            schema: Type[Any],
            type_hints: Optional[dict[str, Any]] = None,
            *,
            max_depth: int = 12,
    ) -> "SchemaCoercionMapper":
        by_depth = _cache.setdefault(schema, {})
        if max_depth in by_depth:
            return by_depth[max_depth]
        inst = super().__new__(cls)
        by_depth[max_depth] = inst
        return inst

    def __init__(
            self,
            schema: Type[Any],
            type_hints: Optional[dict[str, Any]] = None,
            *,
            max_depth: int = 12,
    ) -> None:
        if hasattr(self, "_initialised"):
            return
        self._initialised = True

        self.schema = schema
        self.max_depth = max_depth

        self.type_hints = (
            type_hints
            if type_hints is not None
            else get_type_hints(schema, localns={schema.__name__: schema}, include_extras=True)
        )

        if issubclass(schema, BaseModelV1):
            self._fields = {
                n: self.type_hints.get(n, f.annotation)
                for n, f in schema.__fields__.items()
            }
            self._construct = schema.construct

        elif issubclass(schema, BaseModel):
            self._fields = {
                n: self.type_hints.get(n, f.annotation)
                for n, f in schema.model_fields.items()
            }
            self._construct: Callable[..., Any] = schema.model_construct  # type: ignore

        else:
            raise TypeError("Schema is neither a Pydantic v1 nor v2 model.")

        self._field_coercers: Optional[dict[str, Callable[[Any, int], Any]]] = None

    def __call__(self, input_data: Any, depth: Optional[int] = None) -> Any:
        return self.coerce(input_data, depth)

    def coerce(self, input_data: Any, depth: Optional[int] = None) -> Any:
        if depth is None:
            depth = self.max_depth
        if not isinstance(input_data, dict) or depth <= 0:
            return input_data

        if self._field_coercers is None:
            self._field_coercers = {
                n: self._build_coercer(t, depth - 1) for n, t in self._fields.items()
            }

        processed: dict[str, Any] = {}
        for k, v in input_data.items():
            fn = self._field_coercers.get(k)
            processed[k] = fn(v, depth - 1) if fn else v
        return self._construct(**processed)

    def _build_coercer(
            self, field_type: Any, depth: int, *, throw: bool = False
    ) -> Callable[[Any, Any], Any]:
        if depth == 0:
            return self._passthrough

        # unwrap Annotated
        field_type, metadata = self._unwrap_annotated(field_type)
        origin = get_origin(field_type)

        if (field_type in _IDENTITY_TYPES) or (origin in _IDENTITY_TYPES):
            return self._passthrough

        # support TypeVar
        if isinstance(field_type, TypeVar):
            concrete = self.type_hints.get(field_type)  # type: ignore
            if concrete is not None:
                return self._build_coercer(concrete, depth - 1)
            return self._passthrough

        # support generics like Wrapper[int] Wrapper[AnyMessage]
        if hasattr(field_type, "__parameters__") and hasattr(field_type, "model_fields"):
            try:
                type_hints = self.resolve_concrete_type_hints(field_type)

                def generic_model_coercer(v: Any, d: int) -> Any:
                    if not isinstance(v, dict):
                        if throw:
                            raise TypeError(f"Expected dict for {field_type}, got {type(v)}")
                        return v
                    mapper = SchemaCoercionMapper(field_type, type_hints, max_depth=d)
                    return mapper.coerce(v, d)

                return generic_model_coercer
            except Exception as e:
                logger.debug(f"Generic type resolution failed: {e}")
                return self._passthrough

        if isclass(field_type):
            # This is needed bcs. of issubclass issues on older versions of python
            is_class_ = True
            try:
                is_bm_v2 = issubclass(field_type, BaseModel)
            except TypeError:
                # python < 3.11 issue.
                is_class_ = False
                is_bm_v2 = False
            if is_bm_v2 or (is_class_ and issubclass(field_type, BaseModelV1)):
                mapper = SchemaCoercionMapper(field_type, max_depth=depth - 1)
                return lambda v, d: mapper.coerce(v, d) if isinstance(v, dict) else v

        if origin is list:
            args = get_args(field_type)
            if len(args) != 1:
                return self._passthrough
            sub = self._build_coercer(args[0], depth - 1)

            def list_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, (list, tuple)):
                    return v
                return [sub(x, d - 1) for x in v]

            return list_coercer

        if origin is set or field_type is set:
            args = get_args(field_type)
            if len(args) > 1:
                return self._passthrough
            elif len(args) == 1:
                sub = self._build_coercer(args[0], depth - 1)
            else:
                sub = None  # type: ignore

            def set_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, (list, tuple, set)):
                    return v
                if sub is None:
                    return set(v)
                return {sub(x, d - 1) for x in v}

            return set_coercer

        if origin is dict or field_type is dict:
            args = get_args(field_type)
            if len(args) != 2:

                def dict_coercer(v: Any, d: Any) -> Any:
                    if not isinstance(v, dict):
                        if throw:
                            raise TypeError("Expected dict, got %s" % type(v))
                    return v

                return dict_coercer
            k_sub = self._build_coercer(args[0], depth - 1)
            v_sub = self._build_coercer(args[1], depth - 1)

            def dict_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, dict):
                    if throw:
                        raise TypeError("Expected dict, got %s" % type(v))
                    return v
                return {k_sub(k, d - 1): v_sub(val, d - 1) for k, val in v.items()}

            return dict_coercer

        if origin is tuple:
            elem_types = get_args(field_type)
            if not elem_types:
                return self._passthrough
            subs = [self._build_coercer(t, depth - 1) for t in elem_types]
            return lambda v, d: (
                tuple(
                    subs[i](v[i] if i < len(v) else None, d - 1)
                    for i in range(len(subs))
                )
                if isinstance(v, (list, tuple))
                else v
            )

        if origin is Union:
            args = get_args(field_type)
            discriminator_key = self._extract_discriminator_key(metadata)
            none_in_union = False
            discriminator_map = {}

            for arg in args:
                if arg is type(None):
                    none_in_union = True
                    continue
                base_type = arg
                if get_origin(arg) is Annotated:
                    base_type, _ = get_args(arg)[0], get_args(arg)[1:]
                try:
                    hint = get_type_hints(base_type)
                    lit = hint.get(discriminator_key)
                    if get_origin(lit) is Literal:
                        for val in get_args(lit):
                            discriminator_map[val] = base_type
                except Exception as e:
                    if throw:
                        raise e
                    else:
                        logger.debug(f"Failed to extract discriminator: {e}")

            def union_coercer(v: Any, d: Any) -> Any:
                if v is None and none_in_union:
                    return None

                tag = None
                if callable(discriminator_key):
                    try:
                        tag = discriminator_key(v)
                    except Exception as e:
                        logger.debug(f"Failed to call discriminator func: {e}")
                elif isinstance(v, dict) and isinstance(discriminator_key, str) and discriminator_key in v:
                    tag = v[discriminator_key]

                if tag is not None:
                    for arg in args:
                        base_type = arg
                        if get_origin(arg) is Annotated:
                            base_type, _ = get_args(arg)[0], get_args(arg)[1:]

                        try:
                            if issubclass(base_type, (BaseModel, BaseModelV1)):
                                return SchemaCoercionMapper(base_type, max_depth=d).coerce(v, d)
                        except Exception as e:
                            logger.debug(f"Coercion with {base_type} failed for tag={tag}: {e}")
                            continue

                # fallback: try coercing each branch
                for arg in args:
                    try:
                        sub = self._build_coercer(arg, d - 1)
                        return sub(v, d - 1)
                    except Exception as e:
                        if throw:
                            raise e
                        else:
                            logger.debug(f"Fallback coercion failed for arg={arg}: {e}")

                return v

            return union_coercer

        adapter_fn = _get_adapter(field_type)
        return lambda v, _d: adapter_fn(v)

    @staticmethod
    def _passthrough(v: Any, _d: Any) -> Any:  # noqa: D401
        return v

    @staticmethod
    def _extract_discriminator_key(meta: list[Any]) -> str | Callable[[Any], Hashable]:
        """Extract discriminator field name or function from Annotated metadata"""
        for m in meta:
            if isinstance(m, FieldInfo):
                disc = getattr(m, "discriminator", None)
                if isinstance(disc, Discriminator):
                    return disc.discriminator
                elif isinstance(disc, str):
                    return disc
        return "type"

    @staticmethod
    def _unwrap_annotated(tp: Any) -> tuple[Any, list[Any]]:
        """Unwrap nested Annotated types, extracting the base type and all metadata"""
        metadata = []
        while get_origin(tp) is Annotated:
            tp, *meta = get_args(tp)
            metadata.extend(meta)
        return tp, metadata

    @staticmethod
    def resolve_concrete_type_hints(generic_model_type: Any) -> dict[Any, Any]:
        """Resolve concrete type hints in a generic model"""
        origin = get_origin(generic_model_type)
        args = get_args(generic_model_type)
        param_names = getattr(origin, "__parameters__", [])

        if not args or not param_names:
            return {}

        type_map = dict(zip(param_names, args))
        result = {}

        for field_name, model_field in origin.model_fields.items():
            anno = model_field.annotation
            if get_origin(anno) is Annotated:
                base, *meta = get_args(anno)
                if isinstance(base, TypeVar) and base in type_map:
                    result[field_name] = Annotated[type_map[base], *meta]
                else:
                    result[field_name] = anno
            elif isinstance(anno, TypeVar) and anno in type_map:
                result[field_name] = type_map[anno]
            else:
                result[field_name] = anno

        return result


_adapter_cache: dict[Any, Callable[[Any], Any]] = {}

_IDENTITY_TYPES: tuple[type[Any], ...] = (
    int,
    float,
    str,
    bool,
    bytes,
    bytearray,
    complex,
    memoryview,
    type(None),
)

try:
    # Pydantic v2.
    from pydantic import TypeAdapter

    try:
        import pydantic.v1.types as v1_types_
        from pydantic.v1 import parse_obj_as

        v1_types = tuple(
            v for k, v in vars(v1_types_).items() if k in v1_types_.__all__
        )
    except ImportError:
        v1_types = ()

        def parse_obj_as(tp: Any, v: Any) -> Any:  # type: ignore
            return v

    try:
        from pydantic.v1 import parse_obj_as
        from pydantic.v1.main import create_model
    except ImportError:
        create_model = None  # type: ignore

    def _get_v1_parser(tp: Any) -> Any:
        if create_model is not None:
            try:
                parser = create_model(
                    f"ParsingModel[{tp}]",
                    __root__=(tp, ...),
                )
                return lambda v: parser(__root__=v).__root__  # type: ignore
            except RuntimeError:
                return lambda v: v
        return lambda v: parse_obj_as(tp, v)

    @functools.lru_cache(maxsize=2048)
    def _adapter_for(tp: Any) -> Callable[[Any], Any]:  # noqa: D401
        if tp in v1_types:
            return _get_v1_parser(tp)
        try:
            return TypeAdapter(
                tp, config={"arbitrary_types_allowed": True}
            ).validate_python
        except TypeError:
            # Delayed classes like ConstrainedList
            return _get_v1_parser(tp)

except ImportError:
    # Pydantic V1
    from pydantic.v1.main import create_model

    @functools.lru_cache(maxsize=2048)
    def _adapter_for(tp: Any) -> Callable[[Any], Any]:  # noqa: D401
        try:
            parser = create_model(
                f"ParsingModel[{tp}]",
                __root__=(tp, ...),
            )
            return lambda v: parser(__root__=v).__root__  # type: ignore
        except RuntimeError:
            return lambda v: v


def _get_adapter(tp: Any) -> Callable[[Any], Any]:
    try:
        return _adapter_cache[tp]
    except KeyError:
        fn = _adapter_for(tp)
        _adapter_cache[tp] = fn
        return fn
