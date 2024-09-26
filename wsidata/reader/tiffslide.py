from .openslide import OpenSlideReader


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

    def create_reader(self):
        from tiffslide import TiffSlide

        self._reader = TiffSlide(self.file)
