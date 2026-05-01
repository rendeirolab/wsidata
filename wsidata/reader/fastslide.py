from pathlib import Path
from typing import Union

from ._reader_registry import register
from .base import AssociatedImages, ReaderBase, convert_image


@register(name="fastslide")
class FastSlideReader(ReaderBase):
    """
    Use FastSlide to interface with image files.

    Depends on `fastslide <https://github.com/NKI-AI/fastslide>`_.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "fastslide"
    pkg_namespaces = "fastslide"
    extensions = (
        ".svs",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".tiff",
        ".tif",
        ".ome.tiff",
        ".ome.tif",
        ".ome.zarr",
        ".svslide",
        ".bif",
        ".isyntax",
        ".dcm",
        ".dicom",
        ".czi",
    )

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
        level = self.translate_level(level)
        # All types are coerced to native Python types
        img = self.reader.read_region(
            (int(x), int(y)), int(level), (int(width), int(height))
        )
        return convert_image(img)

    def get_thumbnail(self, size, **kwargs):
        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if sx > sy:
            size = (size, int(size * sy / sx))
        else:
            size = (int(size * sx / sy), size)

        img = self.reader.get_thumbnail(size)
        return convert_image(img)

    def detach_reader(self):
        if self._reader is not None:
            self._reader.close()
            self.set_reader(None)

    def create_reader(self):
        from fastslide import FastSlide

        self._reader = FastSlide.from_file_path(self.file)

    @property
    def associated_images(self):
        """The associated images in a key-value pair"""
        if self._associated_images is None:
            self._associated_images = AssociatedImages(
                {k: v.convert("RGB") for k, v in self._reader.associated_images.items()}
            )
        return self._associated_images
