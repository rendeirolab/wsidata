import json
from math import floor
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
    supports_scenes = True
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
        scene: int | None = None,
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()
        self._select_scene(scene)
        self._set_properties()

    def _select_scene(self, scene):
        associated_names = {
            name.casefold() for name in self._reader.associated_images.keys()
        }
        views = [
            view
            for view in self._reader.images
            if (view.name or "").casefold() not in associated_names
        ]
        if not views:
            views = list(self._reader.images)

        scene_names = [view.name or f"Image {i}" for i, view in enumerate(views)]
        if scene is None:
            primary_index = self._reader.images.primary_index
            scene = next(
                (i for i, view in enumerate(views) if view.index == primary_index),
                max(
                    range(len(views)),
                    key=lambda i: views[i].dimensions[0] * views[i].dimensions[1],
                ),
            )
        scene = self.validate_scene(scene, len(views))
        self._native_scene = views[scene].index
        self._scene = scene
        self._scene_names = scene_names

    @property
    def _scene_view(self):
        # A view dies with the slide that made it, so take it from the open
        # slide: self.reader re-opens a slide closed by detach_reader()
        return self.reader.images[self._native_scene]

    def _set_properties(self):
        reader = self._scene_view
        level_shape = [
            [int(height), int(width)] for width, height in reader.level_dimensions
        ]
        level_downsample = [float(value) for value in reader.level_downsamples]
        mpp_x, mpp_y = reader.mpp
        valid_mpp = [
            value for value in (mpp_x, mpp_y) if value is not None and value > 0
        ]
        magnification = self._reader.properties.get("objective_magnification")
        raw = dict(self._reader.properties)
        raw.update(
            {
                "scene": self._scene,
                "native_scene": reader.index,
                "scene_name": reader.name,
                "n_scenes": len(self._scene_names),
            }
        )

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
                0,
                0,
                int(reader.dimensions[0]),
                int(reader.dimensions[1]),
            ],
            scene=self._scene,
            n_scenes=len(self._scene_names),
            scene_names=self._scene_names,
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
        level = int(self.translate_level(level))
        # fastslide reads in level coordinates, and only inside the level
        downsample = self.properties.level_downsample[level]
        x, y = floor(x / downsample), floor(y / downsample)
        width, height = int(width), int(height)
        level_height, level_width = self.properties.level_shape[level]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + width, level_width), min(y + height, level_height)
        if (x0, y0, x1, y1) == (x, y, x + width, y + height):
            img = self._scene_view.read_region((x, y), level, (width, height))
            return convert_image(img.numpy())
        # Read the part inside the level and leave the rest black, as the
        # other readers do
        region = np.zeros((height, width, 3), dtype=np.uint8)
        if x0 < x1 and y0 < y1:
            img = self._scene_view.read_region((x0, y0), level, (x1 - x0, y1 - y0))
            region[y0 - y : y1 - y, x0 - x : x1 - x] = convert_image(img.numpy())
        return region

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
        level = self._scene_view.get_best_level_for_downsample(downsample)
        level_width, level_height = self._scene_view.level_dimensions[level]
        img = self._scene_view.read_region((0, 0), level, (level_width, level_height))
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
            images = self.reader.associated_images
            self._associated_images = AssociatedImages(
                {
                    key: Image.fromarray(convert_image(images[key].numpy()))
                    for key in images.keys()
                }
            )
        return self._associated_images
