import sys
import types
from typing import Any, Callable, Optional

from langgraph.constants import RETURN
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
from langgraph.utils.runnable import RunnableSeq, coerce_to_runnable

"""
Utilities borrowed from cloudpickle.
https://github.com/cloudpipe/cloudpickle/blob/6220b0ce83ffee5e47e06770a1ee38ca9e47c850/cloudpickle/cloudpickle.py#L265
"""


def _getattribute(obj: Any, name: str) -> Any:
    for subpath in name.split("."):
        if subpath == "<locals>":
            raise AttributeError(
                "Can't get local attribute {!r} on {!r}".format(name, obj)
            )
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError(
                "Can't get attribute {!r} on {!r}".format(name, obj)
            ) from None
    return obj, parent


def _whichmodule(obj: Any, name: str) -> Optional[str]:
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    module_name = getattr(obj, "__module__", None)

    if module_name is not None:
        return module_name
    # Protect the iteration by using a copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr or
    # other threads importing at the same time.
    for module_name, module in sys.modules.copy().items():
        # Some modules such as coverage can inject non-module objects inside
        # sys.modules
        if (
            module_name == "__main__"
            or module_name == "__mp_main__"
            or module is None
            or not isinstance(module, types.ModuleType)
        ):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None


def _lookup_module_and_qualname(
    obj: Any, name: Optional[str] = None
) -> Optional[tuple[types.ModuleType, str]]:
    if name is None:
        name = getattr(obj, "__qualname__", None)
    if name is None:  # pragma: no cover
        # This used to be needed for Python 2.7 support but is probably not
        # needed anymore. However we keep the __name__ introspection in case
        # users of cloudpickle rely on this old behavior for unknown reasons.
        name = getattr(obj, "__name__", None)
    if name is None:
        return None

    module_name = _whichmodule(obj, name)

    if module_name is None:
        # In this case, obj.__module__ is None AND obj was not found in any
        # imported module. obj is thus treated as dynamic.
        return None

    if module_name == "__main__":
        return None

    # Note: if module_name is in sys.modules, the corresponding module is
    # assumed importable at unpickling time. See #357
    module = sys.modules.get(module_name, None)
    if module is None:
        # The main reason why obj's module would not be imported is that this
        # module has been dynamically created, using for example
        # types.ModuleType. The other possibility is that module was removed
        # from sys.modules after obj was created/imported. But this case is not
        # supported, as the standard pickle does not support it either.
        return None

    try:
        obj2, parent = _getattribute(module, name)
    except AttributeError:
        # obj was not found inside the module it points to
        return None
    if obj2 is not obj:
        return None
    return module, name


def get_runnable_for_func(func: Callable[..., Any]) -> RunnableSeq:
    if func in CACHE:
        return CACHE[func]
    else:
        seq = RunnableSeq(
            coerce_to_runnable(func, name=None, trace=False),
            ChannelWrite([ChannelWriteEntry(RETURN)]),
            name=func.__name__,
        )
        if not _lookup_module_and_qualname(func):
            return seq
        return CACHE.setdefault(func, seq)


CACHE: dict[Callable[..., Any], RunnableSeq] = {}
