# Modified from pandas accessor implementation
from __future__ import annotations

import warnings

from wsidata._utils import find_stack_level


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def _register_accessor(name: str, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        # cls.accessors.add(name)
        return accessor

    return decorator


def register_wsidata_accessor(name: str):
    """Register a custom accessor on WSIData objects.

    Examples
    --------

    Create a custom accessor for WSIData objects,
    the init method of the accessor class should accept a single argument,
    which is the WSIData object.

    .. code-block:: python

        >>> @register_wsidata_accessor("my_accessor")
        ... class MyAccessor:
        ...    def __init__(self, obj):
        ...        self.obj = obj
        ...
        ...    def my_method(self):
        ...        return "Hello, world!"


    """
    from .._model import WSIData

    return _register_accessor(name, WSIData)
