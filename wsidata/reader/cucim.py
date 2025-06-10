import warnings
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .._utils import find_stack_level
from .base import AssociatedImages, ReaderBase, SlideProperties, convert_image


class CuCIMReader(ReaderBase):
    """
    Use CuCIM to interface with image files.

    See `CuCIM <https://github.com/rapidsai/cucim>`_ for more information.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    def __init__(
        self,
        file: Union[Path, str],
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()
        self._dtype = None
        self._process_cucim_properties(self.reader)

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
        img = np.asarray(
            self.reader.read_region(
                (x, y),
                (int(width), int(height)),
                level=level,
                **kwargs,
            ),
            dtype=self._dtype,
        )
        return convert_image(img)

    def get_thumbnail(self, size, **kwargs):
        if "thumbnail" in self.associated_images:
            return np.asarray(self.associated_images["thumbnail"])

        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if sx > sy:
            size = (size, int(size * sy / sx))
        else:
            size = (int(size * sx / sy), size)

        img = self.get_level(-1)
        img = Image.fromarray(img).thumbnail(size, Image.Resampling.LANCZOS)
        return convert_image(img)

    def detach_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def create_reader(self):
        from cucim import CuImage

        self._reader = CuImage(self.file)

    def _process_cucim_properties(self, reader):
        shape = reader.shape[0:2]
        bounds = [0, 0, shape[1], shape[0]]
        self._dtype = reader.typestr
        mpp = None
        magnification = None

        # CuCIM metadata
        resolutions = reader.resolutions
        level_shape = resolutions.get("level_dimensions")
        level_downsample = resolutions.get("level_downsamples")
        n_level = resolutions.get("level_count", len(level_shape))
        if level_shape is not None:
            level_shape = [list(dim)[::-1] for dim in level_shape]
        if level_downsample is not None:
            level_downsample = list(level_downsample)

        # Aperio metadata
        aperio_metadata = reader.metadata.get("aperio")
        if aperio_metadata is not None:
            mpp = aperio_metadata.get("MPP")
            magnification = aperio_metadata.get("AppMag")

        self.properties = SlideProperties(
            shape=shape,
            n_level=n_level,
            level_shape=level_shape,
            level_downsample=level_downsample,
            mpp=mpp,
            magnification=magnification,
            bounds=bounds,
            raw=reader.metadata,
        )

    @property
    def reader(self):
        if self._reader is None:
            self.create_reader()
        return self._reader

    @property
    def associated_images(self):
        """The associated images in a key-value pair"""
        if self._associated_images is None:
            images = self.reader.associated_images
            if len(images) == 0:
                self._associated_images = AssociatedImages({})
            else:
                imgs = {}
                for k in images:
                    img_obj = self.reader.associated_image(k)
                    img_shape = img_obj.shape
                    img_shape = img_shape[0:2]  # Get only the first two dimensions
                    try:
                        img_arr = Image.fromarray(np.asarray(img_obj))
                        imgs[k] = img_arr
                    except Exception as e:
                        warnings.warn(
                            f"Failed to convert associated image '{k}' to array: {e}",
                            stacklevel=find_stack_level(),
                        )

                self._associated_images = AssociatedImages(imgs)
        return self._associated_images
