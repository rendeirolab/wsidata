from ._datatree import to_datatree
from ._reader_registry import READERS
from .base import ReaderBase, SlideProperties
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .fastslide import FastSlideReader
from .isyntax import ISyntaxReader
from .kfbslide import KFBSlideReader
from .openslide import OpenSlideReader
from .pylibczi import PylibCZIReader
from .spatialdata_image2d import SpatialDataImage2DReader
from .tiffslide import TiffSlideReader

__all__ = [
    "READERS",
    "ReaderBase",
    "OpenSlideReader",
    "SlideProperties",
    "BioFormatsReader",
    "CuCIMReader",
    "FastSlideReader",
    "ISyntaxReader",
    "KFBSlideReader",
    "PylibCZIReader",
    "SpatialDataImage2DReader",
    "TiffSlideReader",
    "to_datatree",
]
