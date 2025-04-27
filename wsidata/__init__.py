"""Data structures and I/O functions for whole-slide images (WSIs)."""

__version__ = "0.6.0"

import wsidata.dataset as dataset
from ._model import WSIData, TileSpec
from .io import open_wsi, agg_wsi
from .accessors import (
    register_wsidata_accessor,
    FetchAccessor,
    IterAccessor,
    DatasetAccessor,
)
from .reader import get_reader, SlideProperties
from ._normalizer import ColorNormalizer


__all__ = [
    "open_wsi",
    "agg_wsi",
    "dataset",
    "WSIData",
    "TileSpec",
    "register_wsidata_accessor",
    "FetchAccessor",
    "IterAccessor",
    "DatasetAccessor",
    "get_reader",
    "SlideProperties",
    "ColorNormalizer",
]
