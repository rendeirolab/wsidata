from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2

from .base import AssociatedImages, ReaderBase, SlideProperties, convert_image

if TYPE_CHECKING:
    from spatialdata.models import Image2DModel


class SpatialDataImage2DReader(ReaderBase):
    """
    A wrapper around spatialdata to interface with image files.

    Parameters
    ----------
    img: Image2DModle

    """

    name = "spatialdata"

    def __init__(
        self,
        img: "Image2DModel",
        key=None,
        **kwargs,
    ):
        from xarray import DataTree

        self.img = img
        self.file = key
        self.is_multiscale = False
        if isinstance(img, DataTree):
            self.is_multiscale = True
            groups = list(img.groups)[1::]
            n_level = len(groups)
            # Make sure it's 3 channels
            n_channel = len(img["scale0"].c)
            if n_channel != 3:
                raise ValueError(f"Expected RGB channels, got {n_channel} channels")
            # Make sure the channel names are correct
            channel_names = list(img["scale0"].coords.keys())
            if channel_names != ["c", "y", "x"]:
                raise ValueError(
                    f"Expected channel names ['c', 'y', 'x'], got {channel_names}"
                )
            # Make sure the data type is uint8
            for g in groups:
                if img[g].image.dtype != "uint8":
                    raise ValueError(
                        f"Expected uint8 dtype for image, got {img[g].image.dtype},"
                    )
            level_shape = []
            level_downsample = []
            prev = None
            for level in groups:
                s = img[level].image.shape[1::]
                level_shape.append(s)
                if prev is None:
                    level_downsample.append(1)
                    prev = s
                else:
                    level_downsample.append(prev[0] / s[0])
            self._level_keys = dict(zip(range(n_level), groups))
        else:
            n_level = 1
            n_channel = len(img.c)
            if n_channel != 3:
                raise ValueError(f"Expected RGB channels, got {n_channel} channels")
            channel_names = list(img.coords.keys())
            if channel_names != ["c", "y", "x"]:
                raise ValueError(
                    f"Expected channel names ['c', 'y', 'x'], got {channel_names}"
                )
            if img.dtype != "uint8":
                raise ValueError(
                    f"Expected uint8 dtype for image, got {img.dtype}, consider converting to uint8."
                )
            level_shape = [img.shape[1::]]
            level_downsample = [1]
            self._level_keys = {0: None}

        properties = SlideProperties(
            shape=level_shape[0],
            mpp=None,
            magnification=None,
            n_level=n_level,
            level_shape=level_shape,
            level_downsample=level_downsample,
            bounds=[0, 0, level_shape[0][1], level_shape[0][0]],
        )

        self.set_properties(properties)

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
        key = self._level_keys[level]
        if key is not None:
            img = self.img[key]
        else:
            img = self.img

        downsample_factor = self.properties.level_downsample[level]
        # SpatialData coordinates must be translated to the respective level
        x = x / downsample_factor
        y = y / downsample_factor
        width = width / downsample_factor
        height = height / downsample_factor
        if self.is_multiscale:
            data = img.sel(
                y=slice(y, y + height), x=slice(x, x + width)
            ).image.data.compute()
        else:
            data = img.sel(y=slice(y, y + height), x=slice(x, x + width)).data.compute()
        return data.transpose(1, 2, 0)

    def get_thumbnail(self, size, **kwargs):
        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if sx > sy:
            size = (size, int(size * sy / sx))
        else:
            size = (int(size * sx / sy), size)

        # Check which level to return
        level_shape = self.properties.level_shape
        for ix, s in enumerate(level_shape):
            if s[0] >= size[0] and s[1] >= size[1]:
                break
        # Get the image at that level
        img = self.get_level(ix)
        img = cv2.resize(img, size)
        return img

    def detach_reader(self):
        pass

    def create_reader(self):
        pass

    @property
    def associated_images(self):
        return None
