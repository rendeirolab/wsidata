"""Data structures and I/O functions for whole-slide images (WSIs)."""

__version__ = "0.1.0"

import wsidata.dataset as dataset
from ._model import WSIData, TileSpec
from ._io import open_wsi, agg_wsi
from ._accessors import register_wsidata_accessor, GetAccessor, IterAccessor
from ._reader import get_reader
