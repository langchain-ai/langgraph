import logging
import weakref
from inspect import isclass
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


class SchemaCoercionMapper:
    _cache: weakref.WeakKeyDictionary[Type[Any], dict[int, "SchemaCoercionMapper"]] = (
        weakref.WeakKeyDictionary()
    )

    def __new__(cls, schema: Type[Any], max_depth: int = 5) -> "SchemaCoercionMapper":
        if schema not in cls._cache:
            cls._cache[schema] = {}
        if max_depth in cls._cache[schema]:
            return cls._cache[schema][max_depth]

        inst = super().__new__(cls)
        cls._cache[schema][max_depth] = inst
        return inst

    def __init__(self, schema: Type[Any], max_depth: int = 5):
        if hasattr(self, "_inited"):
            return
        self._inited = True
        self.schema = schema
        self.max_depth = max_depth
        if hasattr(schema, "model_fields") and hasattr(schema, "model_construct"):
            self._fields = {n: f.annotation for n, f in schema.model_fields.items()}
            self._construct = schema.model_construct
        elif hasattr(schema, "__fields__") and callable(
            getattr(schema, "construct", None)
        ):
            self._fields = {n: f.annotation for n, f in schema.__fields__.items()}
            self._construct = schema.construct
        else:
            raise TypeError("Schema is neither valid Pydantic v1 nor v2 model.")
        self._field_coercers: Optional[dict[str, Callable[[Any, Any], Any]]] = None

    def __call__(self, input_data: Any, depth: Optional[int] = None) -> Any:
        return self.coerce(input_data, depth)

    def coerce(self, input_data: Any, depth: Optional[int] = None) -> Any:
        if depth is None:
            depth = self.max_depth
        if not isinstance(input_data, dict) or depth <= 0:
            return input_data
        processed = {}
        if self._field_coercers is None:
            self._field_coercers = {
                n: self._build_coercer(t) for n, t in self._fields.items()
            }
        for k, v in input_data.items():
            fn = self._field_coercers.get(k)
            processed[k] = fn(v, depth - 1) if fn else v
        return self._construct(**processed)

    def _build_coercer(self, field_type: Any) -> Callable[[Any, Any], Any]:
        origin = get_origin(field_type)
        if origin is Annotated:
            real_type, *_ = get_args(field_type)
            sub = self._build_coercer(real_type)
            return lambda v, d: sub(v, d)
        if isclass(field_type):
            is_class_ = True
            try:
                is_base_model = issubclass(field_type, BaseModel)
            except TypeError:
                is_class_ = False
                is_base_model = False

            if is_base_model:
                mapper = SchemaCoercionMapper(field_type, self.max_depth)
                return lambda v, d: mapper.coerce(v, d) if isinstance(v, dict) else v
            if is_class_ and issubclass(field_type, BaseModelV1):
                mapper = SchemaCoercionMapper(field_type, self.max_depth)
                return lambda v, d: mapper.coerce(v, d) if isinstance(v, dict) else v
        if origin is list or field_type is list:
            args = get_args(field_type)
            if len(args) != 1:
                return lambda v, d: v
            sub = self._build_coercer(args[0])

            def list_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, (list, tuple)):
                    raise TypeError(f"Expected list, got {type(v).__name__}")
                return [sub(x, d - 1) for x in v]

            return list_coercer
        if origin is dict or field_type is dict:
            args = get_args(field_type)
            if len(args) != 2:

                def plain_dict_coercer(v: Any, d: Any) -> Any:
                    if not isinstance(v, dict):
                        raise TypeError(f"Expected dict, got {type(v).__name__}")
                    return v

                return plain_dict_coercer
            k_sub = self._build_coercer(args[0])
            v_sub = self._build_coercer(args[1])

            def dict_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, dict):
                    raise TypeError(f"Expected dict, got {type(v).__name__}")
                return {k_sub(k, d - 1): v_sub(val, d - 1) for k, val in v.items()}

            return dict_coercer

        if origin is tuple:
            targs = get_args(field_type)
            if not targs:
                return lambda v, d: v
            subs = [self._build_coercer(a) for a in targs]

            def tuple_coercer(v: Any, d: Any) -> Any:
                if not isinstance(v, (list, tuple)):
                    raise TypeError(f"Expected tuple-like, got {type(v).__name__}")
                out = []
                for i, sp in enumerate(subs):
                    out.append(sp(v[i] if i < len(v) else None, d - 1))
                return tuple(out)

            return tuple_coercer
        if origin is Union:
            uargs = get_args(field_type)
            subs, none_in_union = [], False
            for arg in uargs:
                if arg is type(None):
                    none_in_union = True
                else:
                    subs.append(self._build_coercer(arg))

            def union_coercer(v: Any, d: Any) -> Any:
                if v is None and none_in_union:
                    return None
                err = None
                for sp in subs:
                    try:
                        return sp(v, d - 1)
                    except Exception as e:
                        err = e
                if err:
                    raise err
                return v

            return union_coercer
        return lambda v, d: v
