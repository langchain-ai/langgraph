import functools
import logging
import weakref
from inspect import isclass
from typing import (
    Annotated,
    Any,
    Callable,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

__all__ = ["SchemaCoercionMapper"]

logger = logging.getLogger(__name__)


_cache: weakref.WeakKeyDictionary[type[Any], dict[int, "SchemaCoercionMapper"]] = (
    weakref.WeakKeyDictionary()
)


class SchemaCoercionMapper:
    """Lightweight coercion of *dict* → *BaseModel* instances."""

    def __new__(
        cls,
        schema: type[Any],
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
        schema: type[Any],
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
            else get_type_hints(schema, localns={schema.__name__: schema})
        )

        if issubclass(schema, BaseModelV1):
            self._fields = {
                n: self.type_hints.get(n, f.annotation)
                for n, f in schema.__fields__.items()
            }
            self._construct = schema.construct
            unhandled_attrs = (
                "__pre_root_validators__",
                "__post_root_validators__",
                "__validators__",
            )
            if any(getattr(schema, c, None) for c in unhandled_attrs):
                self.coerce: Callable[[Any, Any], Union[BaseModelV1, BaseModel]] = (
                    lambda v, _: schema(**v)
                )
            else:
                self.coerce = self._coerce

        elif issubclass(schema, BaseModel):
            self._fields = {
                n: self.type_hints.get(n, f.annotation)
                for n, f in schema.model_fields.items()
            }
            self._construct: Callable[..., Any] = schema.model_construct  # type: ignore
            unhandled_attrs = ("validators", "field_validators", "root_validators")
            if (decorators := getattr(schema, "__pydantic_decorators__", None)) and any(
                getattr(decorators, attr, None) for attr in unhandled_attrs
            ):
                self.coerce = lambda v, _: schema.model_validate(v)
            else:
                self.coerce = self._coerce

        else:
            raise TypeError("Schema is neither a Pydantic v1 nor v2 model.")

        self._field_coercers: Optional[dict[str, Callable[[Any, int], Any]]] = None

    def __call__(self, input_data: Any, depth: Optional[int] = None) -> Any:
        return self.coerce(input_data, depth)

    def _coerce(self, input_data: Any, depth: Optional[int] = None) -> Any:
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

        origin = get_origin(field_type)

        if (field_type in _IDENTITY_TYPES) or (origin in _IDENTITY_TYPES):
            return self._passthrough

        if origin is Annotated:
            real_type, *_ = get_args(field_type)
            sub = self._build_coercer(real_type, depth - 1)
            return lambda v, d: sub(v, d)

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
                            raise TypeError(f"Expected dict, got {type(v)}")
                    return v

                return dict_coercer
            k_sub = self._build_coercer(args[0], depth - 1)
            v_sub = self._build_coercer(args[1], depth - 1)

            def dict_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, dict):
                    if throw:
                        raise TypeError(f"Expected dict, got {type(v)}")
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
            uargs = get_args(field_type)
            subs, none_in_union = [], False
            for ix, arg in enumerate(uargs):
                if arg is type(None):
                    none_in_union = True
                else:
                    subs.append(
                        self._build_coercer(arg, depth - 1, throw=ix < len(uargs) - 1)
                    )

            def union_coercer(v: Any, d: Any) -> Any:
                if v is None and none_in_union:
                    return None
                err = None
                for sp in subs:
                    try:
                        return sp(v, d - 1)
                    except TypeError as e:
                        err = e
                if err:
                    raise err
                return v

            return union_coercer

        adapter_fn = _get_adapter(field_type)
        return lambda v, _d: adapter_fn(v)

    @staticmethod
    def _passthrough(v: Any, _d: Any) -> Any:  # noqa: D401
        return v


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
