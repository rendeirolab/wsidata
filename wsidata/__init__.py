"""Data structures and I/O functions for whole-slide images (WSIs)."""

from ._version import version

__version__ = version

import wsidata.dataset as dataset

from ._model import TileSpec, WSIData
from ._normalizer import ColorNormalizer
from .accessors import (
    DatasetAccessor,
    FetchAccessor,
    IterAccessor,
    register_wsidata_accessor,
)
from .io import agg_wsi, open_wsi
from .reader import SlideProperties, get_reader

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
