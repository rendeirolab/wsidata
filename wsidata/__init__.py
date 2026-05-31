"""Data structures and I/O functions for whole-slide images (WSIs)."""

from ._version import version

__version__ = version

from ._model import TileRequest, TileSpec, WSIData, shapes2tiles
from .accessors import (
    DatasetAccessor,
    FetchAccessor,
    IterAccessor,
    register_wsidata_accessor,
)
from .io import agg_wsi, concat_feature_anndata, open_wsi
from .reader import READERS, SlideProperties


def __getattr__(name):
    # Lazily expose the `dataset` submodule and `ColorNormalizer` so that
    # `import wsidata` does not eagerly import torch (both pull torch at
    # definition time: torch.utils.data.Dataset subclasses / torch.nn.Module).
    if name == "dataset":
        import importlib

        mod = importlib.import_module(".dataset", __name__)
        globals()[name] = mod
        return mod
    if name == "ColorNormalizer":
        from ._normalizer import ColorNormalizer

        globals()[name] = ColorNormalizer
        return ColorNormalizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["dataset", "ColorNormalizer"])


__all__ = [
    "open_wsi",
    "agg_wsi",
    "concat_feature_anndata",
    "dataset",
    "READERS",
    "WSIData",
    "TileSpec",
    "TileRequest",
    "shapes2tiles",
    "register_wsidata_accessor",
    "FetchAccessor",
    "IterAccessor",
    "DatasetAccessor",
    "SlideProperties",
    "ColorNormalizer",
]
