from ._reader_datatree import to_datatree
from .base import ReaderBase, SlideProperties
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .openslide import OpenSlideReader
from .spatialdata_image2d import SpatialDataImage2DReader
from .tiffslide import TiffSlideReader
from .utils import get_reader, try_reader
