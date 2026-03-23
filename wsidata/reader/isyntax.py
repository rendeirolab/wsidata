import io
import json
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from ._reader_registry import register
from .base import AssociatedImages, ReaderBase, SlideProperties, convert_image


@register(name="isyntax")
class ISyntaxReader(ReaderBase):
    """
    Use pyisyntax to interface with Philips iSyntax whole-slide images.

    Depends on `pyisyntax <https://github.com/anibali/pyisyntax>`_.

    Notes
    -----
    `pyisyntax` reads regions using coordinates in the *target pyramid level*
    reference frame, while wsidata's `read_region` API specifies x/y in the
    level 0 reference frame. This reader converts x/y to the requested level.
    """

    name = "isyntax"
    pkg_namespaces = "isyntax"
    pkgs = ["pyisyntax"]

    def __init__(
        self,
        file: Union[Path, str],
        cache_size: int = 2000,
        **kwargs,
    ):
        self.file = str(file)
        self._cache_size = int(cache_size)
        self.create_reader()
        self._process_isyntax_properties(self.reader)

    def create_reader(self):
        from isyntax import ISyntax

        self.set_reader(ISyntax.open(self.file, cache_size=self._cache_size))

    def detach_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
            finally:
                self.set_reader(None)
                self._associated_images = None

    def _process_isyntax_properties(self, reader):
        # pyisyntax reports (width, height)
        width0, height0 = reader.dimensions
        level_dimensions = reader.level_dimensions  # list[(w, h)]
        level_downsamples = reader.level_downsamples  # list[int]

        level_shape = [[int(h), int(w)] for (w, h) in level_dimensions]
        level_downsample = [float(d) for d in level_downsamples]
        n_level = int(reader.level_count)

        # mpp: keep the average for SlideProperties, preserve both in raw
        mpp_x = float(reader.mpp_x) if reader.mpp_x is not None else None
        mpp_y = float(reader.mpp_y) if reader.mpp_y is not None else None
        mpp = None
        if mpp_x is not None and mpp_y is not None:
            mpp = (mpp_x + mpp_y) / 2.0

        raw = {
            "barcode": getattr(reader, "barcode", None),
            "dimensions": [int(width0), int(height0)],
            "level_dimensions": [[int(w), int(h)] for (w, h) in level_dimensions],
            "level_downsamples": [int(d) for d in level_downsamples],
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "offset_x": int(getattr(reader, "offset_x", 0) or 0),
            "offset_y": int(getattr(reader, "offset_y", 0) or 0),
        }

        self.properties = SlideProperties(
            shape=[int(height0), int(width0)],
            n_level=n_level,
            level_shape=level_shape,
            level_downsample=level_downsample,
            mpp=mpp,
            magnification=None,
            bounds=[0, 0, int(width0), int(height0)],
            raw=json.dumps(raw),
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
        downsample = float(self.properties.level_downsample[level])

        # wsidata specifies x/y at level 0; pyisyntax expects x/y at requested level
        open_x, open_y = int(x / downsample), int(y / downsample)
        open_width, open_height = int(width), int(height)

        img_height, img_width = self.properties.level_shape[level]

        # If x, y is out of bounds, directly return a black image
        if open_x >= img_width or open_y >= img_height:
            return np.zeros((open_height, open_width, 3), dtype=np.uint8)

        clip_width = open_x + open_width - img_width
        clip_height = open_y + open_height - img_height

        # Check if the region is out of bounds
        if clip_width > 0:
            open_width -= clip_width
        else:
            clip_width = 0
        if clip_height > 0:
            open_height -= clip_height
        else:
            clip_height = 0

        img = self.reader.read_region(
            int(open_x),
            int(open_y),
            int(open_width),
            int(open_height),
            level=int(level),
        )
        img = convert_image(img)

        # Fill the clipped region with black
        if clip_width > 0 or clip_height > 0:
            img = np.pad(
                img,
                ((0, int(clip_height)), (0, int(clip_width)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return img

    def get_thumbnail(self, size, **kwargs):
        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")

        # The size is the maximum edge length
        if sx > sy:
            target_hw = (size, int(size * sy / sx))
        else:
            target_hw = (int(size * sx / sy), size)

        # Find a suitable level to minimize work, then resize to target.
        #
        # We want the *coarsest* (lowest-resolution) level that is still
        # at least as large as the requested thumbnail.
        level_shape = self.properties.level_shape
        level = len(level_shape) - 1
        for ix in range(len(level_shape) - 1, -1, -1):
            h, w = level_shape[ix]
            if h >= target_hw[0] and w >= target_hw[1]:
                level = ix
                break

        img = self.get_level(level)
        # cv2 wants (width, height)
        img = cv2.resize(
            img,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_AREA,
        )
        return img

    @property
    def associated_images(self):
        """The associated images in a key-value pair."""
        if self._associated_images is None:
            images = {}

            macro_jpeg = self.reader.read_macro_image_jpeg()
            if macro_jpeg is not None:
                img = Image.open(io.BytesIO(bytes(macro_jpeg)), formats=["JPEG"])
                images["macro"] = img.convert("RGB")

            label_jpeg = self.reader.read_label_image_jpeg()
            if label_jpeg is not None:
                img = Image.open(io.BytesIO(bytes(label_jpeg)), formats=["JPEG"])
                images["label"] = img.convert("RGB")

            self._associated_images = AssociatedImages(images)
        return self._associated_images
