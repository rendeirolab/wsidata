import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ._reader_registry import register
from .base import AssociatedImages, ReaderBase, SlideProperties, convert_image


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
        ".vsi",
    )

    def __init__(
        self,
        file: Union[Path, str],
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()
        self._set_properties()

    def _set_properties(self):
        reader = self._reader
        level_shape = [
            [int(height), int(width)] for width, height in reader.level_dimensions
        ]
        level_downsample = [float(value) for value in reader.level_downsamples]
        (bounds_x, bounds_y), (bounds_width, bounds_height) = reader.bounds
        mpp_x, mpp_y = reader.mpp
        valid_mpp = [
            value for value in (mpp_x, mpp_y) if value is not None and value > 0
        ]
        magnification = reader.properties.get("objective_magnification")

        self.properties = SlideProperties(
            shape=level_shape[0],
            n_level=int(reader.level_count),
            level_shape=level_shape,
            level_downsample=level_downsample,
            mpp=float(np.mean(valid_mpp)) if valid_mpp else None,
            magnification=(
                float(magnification)
                if magnification is not None and magnification > 0
                else None
            ),
            bounds=[
                int(bounds_x),
                int(bounds_y),
                int(bounds_width),
                int(bounds_height),
            ],
            raw=json.dumps(dict(reader.properties)),
        )

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
        x, y = self.reader.convert_level0_to_level_native(int(x), int(y), level)
        img = self.reader.read_region(
            (int(x), int(y)), int(level), (int(width), int(height))
        )
        return convert_image(img.numpy())

    def get_thumbnail(self, size, **kwargs):
        height, width = self.properties.shape
        if size > height or size > width:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if height > width:
            size = (int(size * width / height), size)
        else:
            size = (size, int(size * height / width))

        target_size = size
        downsample = max(width / target_size[0], height / target_size[1])
        level = self.reader.get_best_level_for_downsample(downsample)
        level_width, level_height = self.reader.level_dimensions[level]
        img = self.reader.read_region((0, 0), level, (level_width, level_height))
        thumbnail = Image.fromarray(convert_image(img.numpy()))
        thumbnail.thumbnail(target_size, Image.Resampling.LANCZOS)
        return np.asarray(thumbnail)

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
            images = self._reader.associated_images
            self._associated_images = AssociatedImages(
                {
                    key: Image.fromarray(convert_image(images[key].numpy()))
                    for key in images.keys()
                }
            )
        return self._associated_images
