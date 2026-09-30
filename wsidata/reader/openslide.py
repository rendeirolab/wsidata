from pathlib import Path
from typing import Union

import numpy as np

from ._reader_registry import register
from .base import AssociatedImages, ReaderBase, convert_image


@register(name="openslide")
class OpenSlideReader(ReaderBase):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_
    which wraps the `openslide <https://openslide.org/>`_ C library.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "openslide"
    pkg_namespaces = "openslide"
    pkgs = ["openslide-python", "openslide-bin"]
    extensions = (
        ".svs",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".tiff",
        ".tif",
        ".svslide",
        ".bif",
        ".czi",
        ".dcm",
        ".dicom",
    )
    # OpenSlide paints a read in pieces of at most 4096 x 4096 px
    # (openslide_read_region); None reads every region in one call
    _chunk_px = 4096

    def __init__(
        self,
        file: Union[Path, str],
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()
        self.set_properties(self._reader.properties)

    def get_region(
        self,
        x,
        y,
        width,
        height,
        level: int = 0,
        **kwargs,
    ):
        level = int(self.translate_level(level))
        # All types are coerced to native Python types
        x, y, width, height = int(x), int(y), int(width), int(height)
        chunk = self._chunk_px
        if chunk is None or (width <= chunk and height <= chunk):
            img = self.reader.read_region((x, y), level, (width, height))
            return convert_image(img)
        # Converting a read to RGB holds several copies of it, ~16 bytes per px,
        # so read OpenSlide's pieces one at a time. Each piece is painted at its
        # own fractional offset in the level: offsets computed as OpenSlide
        # computes them give the same pixels as one read.
        ds = self.reader.level_downsamples[level]
        out = np.empty((height, width, 3), dtype=np.uint8)
        for row in range(0, height, chunk):
            for col in range(0, width, chunk):
                piece = self.reader.read_region(
                    (int(x + col * ds), int(y + row * ds)),
                    level,
                    (min(chunk, width - col), min(chunk, height - row)),
                )
                out[row : row + chunk, col : col + chunk] = convert_image(piece)
        return out

    def get_thumbnail(self, size, **kwargs):
        height, width = self.properties.shape
        if size > height or size > width:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if height > width:
            size = (int(size * width / height), size)
        else:
            size = (size, int(size * height / width))

        img = self.reader.get_thumbnail(size)
        return convert_image(img)

    def detach_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
                self.set_reader(None)
            # There is a chance that the pointer
            # to C-library is already collected
            except TypeError:
                pass

    def create_reader(self):
        from openslide import OpenSlide

        self.set_reader(OpenSlide(self.file))

    @property
    def associated_images(self):
        """The associated images in a key-value pair"""
        if self._associated_images is None:
            self._associated_images = AssociatedImages(
                {k: v.convert("RGB") for k, v in self.reader.associated_images.items()}
            )
        return self._associated_images
