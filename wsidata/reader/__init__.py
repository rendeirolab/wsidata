from importlib.metadata import version

# Decide which datatree implementation to use
from packaging.version import Version

from .base import ReaderBase, SlideProperties
from .bioformats import BioFormatsReader
from .cucim import CuCIMReader
from .openslide import OpenSlideReader
from .spatialdata_image2d import SpatialDataImage2DReader
from .tiffslide import TiffSlideReader
from .utils import get_reader, try_reader

zarr_version = Version(version("zarr"))
if zarr_version.major >= 3:
    from ._reader_datatree_zarr_v3 import to_datatree
else:
    from ._reader_datatree_zarr_v2 import to_datatree
