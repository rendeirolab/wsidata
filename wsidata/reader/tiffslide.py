from ._reader_registry import register
from .openslide import OpenSlideReader


@register(name="tiffslide")
class TiffSlideReader(OpenSlideReader):
    """
    Use TiffSlide to interface with image files.

    Depends on `tiffslide <https://github.com/Bayer-Group/tiffslide>`_.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "tiffslide"
    pkg_namespaces = "tiffslide"

    def create_reader(self):
        from tiffslide import TiffSlide

        self._reader = TiffSlide(self.file)
