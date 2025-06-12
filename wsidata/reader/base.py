from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, dataclass, field
from functools import cached_property, singledispatch
from typing import Dict, List, Mapping, Optional

import cv2
import numpy as np
from PIL import Image

# AnnData cannot serialize Tuple
SHAPE = List[int]


@dataclass
class SlideProperties:
    """
    The properties of the slide

    Attributes
    ----------
    shape : (height, width)
        The shape of the slide
    n_level : int
        The number of pyramids levels
    level_shape : array of (height, width)
        The shape of each pyramid level
    level_downsample : array fo float
        The downsample factor of each pyramid level
    mpp : float
        The physical size of each pixel in microns
    magnification : float
        The magnification of the slide
    bounds : (x, y, width, height)
        The bounds of the slide, in the format [x, y, width, height]
        This is the region of the slide that contains tissue
    raw : dict
        The raw metadata in serialized json format

    """

    shape: SHAPE
    n_level: int
    level_shape: List[SHAPE]
    level_downsample: List[float]
    mpp: Optional[float] = None
    magnification: Optional[float] = None
    bounds: Optional[SHAPE] = None
    raw: Optional[str] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(self, metadata: Mapping):
        """Create SlideProperties from a mapping"""
        metadata = parse_metadata(metadata)
        return SlideProperties(**metadata)

    def to_dict(self):
        """Convert the properties to a dictionary"""
        return asdict(self)

    def to_json(self):
        """Convert the properties to a json string"""
        return json.dumps(asdict(self))

    def _repr_html_(self):
        rows = []
        for k, v in self.to_dict().items():
            if k != "raw":
                rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")

        return (
            "<h4>Slide Properties</h4><table><tr><th>Field</th><th>Value</th></tr>"
            + "".join(rows)
            + "</table>"
        )


class AssociatedImages:
    """A class to hold associated images

    This class is used to hold the associated images in a key-value pair
    where the key is the name of the image and the value is the image data.

    Attributes
    ----------
    images : Dict[str, np.ndarray]
        The associated images in a key-value pair

    """

    def __init__(self, images: Optional[Dict[str, Image.Image]] = None):
        self._images = images or {}
        self._base64_image_store = {}

    def __getitem__(self, key: str) -> np.ndarray:
        """Get the image by key"""
        if key in self._images:
            return np.asarray(self._images[key], dtype=np.uint8)
        raise KeyError(f"Image '{key}' not found in associated images.")

    def __getattr__(self, key: str) -> np.ndarray:
        """Get the image by attr"""
        if key in self._images:
            return self[key]
        raise AttributeError(f"Image '{key}' not found in associated images.")

    def __iter__(self):
        return iter(self._images)

    def __len__(self):
        return len(self._images)

    def __repr__(self):
        return f"AssociatedImages({list(self._images.keys())})"

    def __contains__(self, key: str) -> bool:
        return key in self._images

    def _base64_image(self, key) -> str:
        """Generate base64 for the image, each image is resized to max 128 pixels in width or height"""
        img = self._images[key]
        if key in self._base64_image_store:
            return self._base64_image_store[key]

        # Resize the image to max 128 pixels in width or height
        max_size = 128
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_str = f"data:image/png;base64,{base64_str}"
        self._base64_image_store[key] = base64_str
        return base64_str

    def _repr_html_(self):
        rows = []
        # sort the keys to have a consistent order
        for key in sorted(self._images.keys()):
            base64_str = self._base64_image(key)
            rows.append(
                f"<tr><td>{key}</td><td><img src='{base64_str}' style='max-width: 200px;'></td></tr>"
            )
        return (
            "<h4>Associated Images</h4><table><tr><th>Name</th><th>Image</th></tr>"
            + "".join(rows)
            + "</table>"
        )


class ReaderBase:
    """The base class for all reader

    This class defines the basic interface for all reader

    Attributes
    ----------
    file : str
        The path to the image file
    properties : :class:`SlideProperties <wsidata.SlideProperties>`
        The properties of the slide
    name : str
        The name of the reader
    reader : Any
        The reader object

    """

    file: str
    properties: SlideProperties
    name = "base"
    _reader = None
    _associated_images: AssociatedImages = None

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.file}')"

    def __del__(self):
        self.detach_reader()

    @cached_property
    def _level_mapper(self):
        levels = {}
        n_level = self.properties.n_level
        for lv in np.arange(n_level):
            levels[lv] = lv
            levels[-n_level + lv] = lv
        return levels

    @cached_property
    def raw_properties(self):
        """Get the raw properties of the slide"""
        return json.loads(self.properties.raw)

    @property
    def associated_images(self) -> AssociatedImages | Mapping:
        """The associated images in a key-value pair"""
        if self._associated_images is None:
            return {}
        return self._associated_images

    def translate_level(self, level):
        """Translate the level to the actual level

        level -1 refer to the lowest resolution level

        Parameters
        ----------
        level : int
            The level to translate

        """
        req_level = self._level_mapper.get(level)
        if req_level is None:
            raise ValueError(f"Level {level} doesn't exist.")
        return req_level

    def get_region(self, x, y, width, height, level=0, **kwargs):
        """Get a region from image with top-left corner
        This should return a numpy array in xyc format
        """
        raise NotImplementedError

    def get_center(self, x, y, width, height, level=0, **kwargs):
        """Get a patch from image with center"""
        x -= width / 2
        y -= height / 2
        return self.get_region(x, y, width, height, level=level, **kwargs)

    def get_thumbnail(self, size, **kwargs):
        """Get a thumbnail of the image"""
        raise NotImplementedError

    def get_level(self, level, in_bounds=False) -> np.ndarray[np.uint8]:
        """
        Extract the image at the given level

        A very expensive operation, use with caution

        Parameters
        ----------
        level : int
            The level to extract
        in_bounds : bool
            If True, only extract the region that are in the bounds

        """
        height, width = self.properties.level_shape[level]
        if in_bounds:
            x, y, h, w = self.properties.bounds
        else:
            x, y, h, w = 0, 0, height, width
        return self.get_region(x, y, w, h, level=level)

    def set_properties(self, properties: SlideProperties | Mapping):
        if isinstance(properties, SlideProperties):
            self.properties = properties
        else:
            self.properties = SlideProperties.from_mapping(properties)

    def create_reader(self):
        """Create the reader

        This function should be implemented in the subclass

        1. create the reader
        2. assign the reader to self._reader

        """
        # The basic fallback implementation to create reader
        raise NotImplementedError

    def detach_reader(self):
        """Detach the reader

        This function will be called when the object is deleted or
        sent to other process

        Please implement this function in the subclass

        1. Close the reader
        2. Set self._reader to None
        3. Clean up any resources that are not python-managed

        """
        raise NotImplementedError

    @staticmethod
    def resize_img(
        img: np.ndarray,
        dsize: SHAPE = None,
        scale: float = None,
    ):
        dim = np.asarray(img.shape)
        if dsize is not None:
            return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        if scale is not None:
            dim = np.array(dim * scale, dtype=int)
            return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return None

    @property
    def reader(self):
        if self._reader is None:
            self.create_reader()
        return self._reader


@singledispatch
def convert_image(img):
    raise NotImplementedError(f"Unsupported type {type(img)}")


@convert_image.register(Image.Image)
def _(img: Image.Image):
    return cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGBA2RGB)


@convert_image.register(np.ndarray)
def _(img: np.ndarray):
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2RGB)


MAG_KEY = "objective-power"
MPP_KEYS = ("mpp-x", "mpp-y")

LEVEL_HEIGHT_KEY = lambda level: f"level[{level}].height"  # noqa: E731
LEVEL_WIDTH_KEY = lambda level: f"level[{level}].width"  # noqa: E731
LEVEL_DOWNSAMPLE_KEY = lambda level: f"level[{level}].downsample"  # noqa: E731


def parse_metadata(metadata: Mapping):
    metadata = dict(metadata)
    new_metadata = {}
    for k, v in metadata.items():
        new_metadata[".".join(k.split(".")[1::])] = v
    metadata.update(new_metadata)

    fields = set(metadata.keys())

    mpp_keys = []
    # openslide specific mpp keys
    if MPP_KEYS[0] in fields:
        mpp_keys.append(MPP_KEYS[0])
    if MPP_KEYS[1] in fields:
        mpp_keys.append(MPP_KEYS[1])
    for k in fields:
        # Any keys end with .mpp
        if k.lower().endswith("mpp"):
            mpp_keys.append(k)

    mpp = None
    for k in mpp_keys:
        mpp_tmp = metadata.get(k)
        if mpp_tmp is not None:
            mpp = float(mpp_tmp)

    # search magnification
    # search other available mpp keys
    mag_keys = []
    if MAG_KEY in fields:
        mag_keys.append(MAG_KEY)
    for k in fields:
        # Any keys end with .mpp
        if k.lower().endswith("appmag"):
            mag_keys.append(k)

    mag = None
    for k in mag_keys:
        mag_tmp = metadata.get(k)
        if mag_tmp is not None:
            mag = float(mag_tmp)

    level_shape = []
    level_downsample = []

    # Get the number of levels
    n_level = 0
    while f"level[{n_level}].width" in fields:
        n_level += 1

    for level in range(n_level):
        height = metadata.get(LEVEL_HEIGHT_KEY(level))
        width = metadata.get(LEVEL_WIDTH_KEY(level))
        downsample = metadata.get(LEVEL_DOWNSAMPLE_KEY(level))

        level_shape.append((int(height), int(width)))

        if downsample is not None:
            downsample = float(downsample)
        level_downsample.append(downsample)

    shape = level_shape[0]

    # Get the bounds
    bounds_x = metadata.get("bounds-x")
    bounds_y = metadata.get("bounds-y")
    bounds_height = metadata.get("bounds-height")
    bounds_width = metadata.get("bounds-width")
    # The key may still exist but the value is None
    # So we need to check if the value is None
    bounds_x = 0 if bounds_x is None else int(bounds_x)
    bounds_y = 0 if bounds_y is None else int(bounds_y)
    bounds_height = shape[0] if bounds_height is None else int(bounds_height)
    bounds_width = shape[1] if bounds_width is None else int(bounds_width)
    bounds = [bounds_x, bounds_y, bounds_width, bounds_height]

    metadata = {
        "mpp": mpp,
        "magnification": mag,
        "shape": list(shape),
        "n_level": n_level,
        "level_shape": [list(i) for i in level_shape],
        "level_downsample": level_downsample,
        "raw": json.dumps(metadata),
        "bounds": bounds,
    }

    return metadata
