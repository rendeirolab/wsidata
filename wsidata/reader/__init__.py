from importlib.metadata import version

# Decide which datatree implementation to use
from packaging.version import Version

from ._reader_registry import READERS
from .base import ReaderBase, SlideProperties
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .fastslide import FastSlideReader
from .isyntax import ISyntaxReader
from .openslide import OpenSlideReader
from .pylibczi import PylibCZIReader
from .spatialdata_image2d import SpatialDataImage2DReader
from .tiffslide import TiffSlideReader

zarr_version = Version(version("zarr"))
if zarr_version.major >= 3:
    from ._reader_datatree_zarr_v3 import to_datatree
else:
    from ._reader_datatree_zarr_v2 import to_datatree

__all__ = [
    "version",
    "Version",
    "READERS",
    "ReaderBase",
    "OpenSlideReader",
    "SlideProperties",
    "BioFormatsReader",
    "CuCIMReader",
    "FastSlideReader",
    "ISyntaxReader",
    "PylibCZIReader",
    "SpatialDataImage2DReader",
    "TiffSlideReader",
    "to_datatree",
]
