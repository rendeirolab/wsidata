from .base import ReaderBase, SlideProperties
from .openslide import OpenSlideReader
from .tiffslide import TiffSlideReader
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .utils import get_reader
from ._reader_datatree import to_datatree
