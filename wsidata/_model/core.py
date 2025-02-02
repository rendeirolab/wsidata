from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from functools import cached_property
from numbers import Integral
from pathlib import Path
from typing import Literal, Generator, Sequence

import numpy as np
from PIL.Image import Image, fromarray
from anndata import AnnData
from ome_zarr.io import parse_url
from spatialdata import SpatialData
from spatialdata.models import SpatialElement

from ..accessors import FetchAccessor, IterAccessor, DatasetAccessor
from .._utils import find_stack_level
from ..reader import ReaderBase, SlideProperties


class WSIData(SpatialData):
    """
    A container class combining :class:`SpatialData <spatialdata.SpatialData>`
    and a whole slide image reader.

    .. note::
       Use the :func:`open_wsi` function to create a WSIData object.

    By default, the whole slide image is not attached to the SpatialData.
    A thumbnail version of the whole slide image is attached for visualization purpose.

    .. list-table::
       :header-rows: 1

       * - **Content**
         - **Default key**
         - **Slot**
         - **Type**

       * - Whole slide image
         - :bdg-info:`wsi_thumbnail`
         - :bdg-danger:`images`
         - :class:`DataArray <xarray.DataArray>` (c, y, x) format

       * - Slide Properties
         - :bdg-info:`slide_properties`
         - :bdg-danger:`attrs`
         - :class:`SlideProperties <wsidata.reader.SlideProperties>`

       * - Tissue contours
         - :bdg-info:`tissues`
         - :bdg-danger:`shapes`
         - :class:`GeoDataFrame <geopandas.GeoDataFrame>`

       * - Tile locations
         - :bdg-info:`tiles`
         - :bdg-danger:`shapes`
         - :class:`GeoDataFrame <geopandas.GeoDataFrame>`

       * - Tile specifications
         - :bdg-info:`tile_spec`
         - :bdg-danger:`attrs`
         - :class:`TileSpec <wsidata.TileSpec>`

       * - Features
         - :bdg-info:`{feature_key}_{tile_key}`
         - :bdg-danger:`tables`
         - :class:`AnnData <anndata.AnnData>`


    You can interact with WSIData using the following accessors:

    - :class:`fetch <wsidata.FetchAccessor>`: Access data from the WSIData object.
    - :class:`iter <wsidata.IterAccessor>`: Iterate over data in the WSIData object.
    - :class:`ds <wsidata.DatasetAccessor>`: Create deep learning datasets from the WSIData object.
    - To implement your own accessors, use :func:`register_wsidata_accessor <wsidata.register_wsidata_accessor>`.

    For analysis purpose, you can override two slide properties:

    - **microns per pixel (mpp)**: Using the :meth:`set_mpp` method.
    - **bounds**: Using the :meth:`set_bounds` method.

    Other Parameters
    ----------------
    reader : :class:`ReaderBase <wsidata.reader.ReaderBase>`
        A reader object that can interface with the whole slide image file.
    slide_properties_source : {'slide', 'sdata'}, default: 'sdata'
        The source of the slide properties.

        - "slide": load from the reader object.
        - "sdata": load from the SpatialData object.

    Attributes
    ----------
    properties : :class:`SlideProperties <wsidata.reader.SlideProperties>`
        The properties of the whole slide image.
    reader : :class:`ReaderBase <wsidata.reader.ReaderBase>`
        The reader object for interfacing with the whole slide image.
    wsi_store :
        The store path for the whole slide image.

    """

    TILE_SPEC_KEY = "tile_spec"
    SLIDE_PROPERTIES_KEY = "slide_properties"

    @classmethod
    def from_spatialdata(cls, sdata, reader=None, **kws):
        d = cls(
            images=sdata.images,
            labels=sdata.labels,
            shapes=sdata.shapes,
            tables=sdata.tables,
            points=sdata.points,
            reader=reader,
            **kws,
        )
        d.path = sdata.path
        return d

    def __init__(
        self,
        images=None,
        labels=None,
        shapes=None,
        tables=None,
        points=None,
        attrs=None,
        reader: ReaderBase = None,
        slide_properties_source: Literal["slide", "sdata"] = "sdata",
    ):
        self._reader = reader
        self._wsi_store = None
        self.slide_properties_source = slide_properties_source
        self._exclude_elements = set()
        super().__init__(
            images=images,
            labels=labels,
            shapes=shapes,
            tables=tables,
            points=points,
            attrs=attrs,
        )

        if self.SLIDE_PROPERTIES_KEY not in self:
            self.attrs[self.SLIDE_PROPERTIES_KEY] = reader.properties.to_dict()
        else:
            # Try to load the slide properties from the spatial data
            if slide_properties_source == "slide":
                reader_properties = self.attrs[self.SLIDE_PROPERTIES_KEY]
                if reader_properties != reader.properties.to_dict():
                    # Update the reader properties
                    reader.properties.from_mapping(reader_properties)
                    warnings.warn(
                        "Slide properties in the spatial data is different from the reader properties.",
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )

    def __repr__(self):
        return (
            f"WSI: {self.reader.file}\nReader: {self.reader.name}\n{super().__repr__()}"
        )

    def set_exclude_elements(self, elements):
        """Set the elements to be excluded from serialize to the WSIData object on disk."""
        self._exclude_elements.update(elements)

    def set_wsi_store(self, store: str | Path):
        """Set the on disk path for the WSIData."""
        self._wsi_store = Path(store)

    def _gen_elements(
        self, include_table: bool = False
    ) -> Generator[tuple[str, str, SpatialElement | AnnData], None, None]:
        for i in super()._gen_elements(include_table):
            if i[1] not in self._exclude_elements:
                yield i

    def close(self):
        """Close the reader object."""
        self.reader.detach_reader()

    @property
    def reader(self):
        return self._reader

    @property
    def properties(self) -> SlideProperties:
        return self.reader.properties

    @property
    def wsi_store(self):
        return self._wsi_store

    @property
    def thumbnail(self):
        return self.get_thumbnail(size=500, as_array=False)

    @property
    def name(self):
        return Path(self.reader.file).name

    def tile_spec(self, key: str) -> TileSpec:
        """
        Get the :class:`TileSpec` for a collection of tiles.

        Parameters
        ----------
        key : str
            The key of the tiles.

        """
        if self.TILE_SPEC_KEY in self.attrs:
            spec = self.attrs[self.TILE_SPEC_KEY][key]
            return TileSpec(**spec)

    def set_mpp(self, mpp):
        """Set the microns per pixel (mpp) of the whole slide image."""
        self.properties.mpp = mpp
        self.attrs[self.SLIDE_PROPERTIES_KEY]["mpp"] = mpp

    def set_bounds(self, bounds: tuple[int, int, int, int]):
        """Set the bounds of the whole slide image.

        Parameters
        ----------
        bounds : tuple[int, int, int, int]
            The bounds of the whole slide image in the format [x, y, width, height].

        """
        self.properties.bounds = bounds
        self.tables[self.SLIDE_PROPERTIES_KEY].uns["bounds"] = bounds

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int = 0,
        **kwargs,
    ) -> np.ndarray[np.uint8]:
        """Read a region from the whole slide image.

        Parameters
        ----------
        x : int
            The x-coordinate at level 0.
        y : int
            The y-coordinate at level 0.
        width : int
            The width of the region in pixels.
        height : int
            The height of the region in pixels.
        level : int, default: 0
            The pyramid level.

        """
        if self.reader is None:
            raise ValueError("The reader object is not attached.")
        return self.reader.get_region(x, y, width, height, level=level, **kwargs)

    def get_thumbnail(self, size=500, as_array=False) -> np.ndarray[np.uint8] | Image:
        """Get the thumbnail of the whole slide image.

        Parameters
        ----------
        as_array : bool, default: False
            Return as numpy array.

        """
        img = self.reader.get_thumbnail(size=size)
        if as_array:
            return img
        else:
            return fromarray(img)

    def write(
        self,
        file_path=None,
        overwrite: bool = True,
        consolidate_metadata: bool = True,
        format=None,
    ):
        if file_path is not None:
            file_path = Path(file_path)
            if self.path is None:
                self.path = file_path
        else:
            file_path = self._wsi_store
        super().write(
            file_path=file_path,
            overwrite=overwrite,
            consolidate_metadata=consolidate_metadata,
            format=format,
        )

    def _validate_can_safely_write_to_path(
        self,
        file_path: str | Path,
        overwrite: bool = False,
        saving_an_element: bool = False,
    ) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not isinstance(file_path, Path):
            raise ValueError(
                f"file_path must be a string or a Path object, type(file_path) = {type(file_path)}."
            )

        if Path(file_path).exists():
            if parse_url(file_path, mode="r") is None:
                raise ValueError(
                    "The target file path specified already exists, and it has been detected to not be a Zarr store. "
                    "Overwriting non-Zarr stores is not supported to prevent accidental data loss."
                )
            if not overwrite:
                raise ValueError(
                    "The Zarr store already exists. Use `overwrite=True` to try overwriting the store."
                    "Please note that only Zarr stores not currently in used by the current SpatialData object can be "
                    "overwritten."
                )
        # Skip the workaround for now

    def _check_feature_key(self, feature_key, tile_key=None):
        msg = f"{feature_key} doesn't exist"
        if feature_key in self:
            return feature_key
        else:
            if tile_key is not None:
                feature_key = f"{feature_key}_{tile_key}"
                if feature_key in self:
                    return feature_key
                msg = f"Neither {feature_key} or {feature_key}_{tile_key} exist"

        raise KeyError(msg)

    # Define default accessors as property,
    # this will enable auto-completion in IDE
    # For extension, user can define their own accessors
    # using register_wsidata_accessor
    @cached_property
    def fetch(self):
        return FetchAccessor(self)

    @cached_property
    def iter(self):
        return IterAccessor(self)

    @cached_property
    def ds(self):
        return DatasetAccessor(self)


@dataclass
class TileSpec:
    """Data class for storing tile specifications.

    To enable efficient tils generation, there are 3 levels of tile size:

    1. The destined tile size and level requested by the user
    2. The tile size and level used by the image reader to optimize the performance
    3. The actual tile size at the level 0

    This enables user to request tile size and level that are not exist in the image pyramids.
    For example, if our slide is mpp=0.5 (20X) with pyramids of mpp=[0.5, 2.0, 4.0],
    and user request mpp=1.0 (10X) with tile size 512x512. There is no direct level to read from,
    we need to read from mpp=0.5 with tile size of 1024x1024 and downsample to 512x512.


    Parameters
    ----------
    height : int
        The height of the tile.
    width : int
        The width of the tile.
    stride_height : int
        The height of the stride.
    stride_width : int
        The width of the stride.
    mpp : float, default: None
        The requested microns per pixel of tiles.
    ops_level : int, default: 0
        The level of the tiling operation.
    ops_downsample : float, default: 1
        The downsample factor to transform the operation level
        tiles to the requested level.
    tissue_name : str, optional
        The name of the tissue.

    Attributes
    ----------
    ops_{height, width} : int
        The height/width of the tile when retrieving images.
    ops_stride_{height, width}: int
        The height/width of the stride when retrieving images.
    base_{height, width} : int
        The height/width of the tile at the level 0.
    base_stride_{height, width} : int
        The height/width of the stride at the level 0.

    """

    height: int
    width: int
    stride_height: int
    stride_width: int

    mpp: float | None = None

    ops_level: int = 0  # level of the tiling operation
    ops_downsample: float = 1  # downsample to requested level

    base_level: int = 0  # level of the base tile, always 0
    base_downsample: float = 1.0  # downsample to requested level

    tissue_name: str | None = None

    def __post_init__(self):
        # coerce attributes to correct type
        self.height = int(self.height)
        self.width = int(self.width)
        self.stride_height = int(self.stride_height)
        self.stride_width = int(self.stride_width)
        self.ops_level = int(self.ops_level)
        self.ops_downsample = float(self.ops_downsample)
        self.base_level = int(self.base_level)
        self.base_downsample = float(self.base_downsample)

    @classmethod
    def from_wsidata(
        cls,
        wsi: WSIData,
        tile_px: int | (int, int),
        stride_px: int | (int, int) = None,
        mpp=None,
        ops_level=None,
        slide_mpp=None,
        tissue_name=None,
    ):
        """Create a TileSpec from a WSIData object.

        To tile from the whole slide image, the user needs to specify the tile size and stride size.
        mpp only need to be specified if the user wants to make sure
        the tile size is harmonized across different slides.

        If ops_level is not specified, the optimal level will be calculated based on the requested mpp
        to maximize the performance.

        """
        # Check if the tile size is valid
        tile_w, tile_h = _check_width_height("tile_px", tile_px)

        # Check if the stride size is valid
        stride_w, stride_h = _check_width_height(
            "stride_px", stride_px, default_w=tile_w, default_h=tile_h
        )

        # If user does not override slide mpp, use the recorded slide mpp
        if slide_mpp is None:
            slide_mpp = wsi.properties.mpp

        # If user does not specify mpp, default to ops level is 0
        if mpp is None:
            mpp = wsi.properties.mpp
            if ops_level is not None:
                raise ValueError("Please specify mpp if ops_level is specified.")
            ops_level = 0
            ops_downsample = 1
        # Or if user specify mpp, but the slide mpp is not available, use the default level 0
        elif slide_mpp is None:
            if ops_level is None:
                ops_level = 0
                ops_downsample = 1
            else:
                ops_downsample = wsi.properties.level_downsample[ops_level]

            warning_text = f"Slide mpp is not available, using level {ops_level}."
            # If user specify the mpp at the same time but slide mpp is not available
            # we will inform the user that the mpp will be ignored
            if mpp is not None:
                warning_text += f" Requested mpp={mpp} will be ignored."
            warnings.warn(warning_text, stacklevel=find_stack_level())
        else:
            # If user didn't specify ops level but mpp, an optimized ops level will be calculated
            wsi_downsample = np.asarray(wsi.properties.level_downsample)
            req_downsample = mpp / slide_mpp
            if req_downsample < 1:
                raise ValueError(
                    f"Requested mpp={mpp} is smaller than the slide mpp={slide_mpp}. "
                    f"Up-sampling is not supported."
                )

            gap = wsi_downsample - req_downsample
            gap[gap > 0] = np.inf

            suggest_level = np.argmin(np.abs(gap))
            suggest_downsample = mpp / (slide_mpp * wsi_downsample[suggest_level])
            if ops_level is None:
                # Apply the optimized level
                ops_level = suggest_level
                ops_downsample = suggest_downsample
            else:
                # check if user specified ops level is valid
                ops_level = wsi.reader.translate_level(ops_level)
                ops_downsample = mpp / (
                    slide_mpp * wsi.properties.level_downsample[ops_level]
                )
                if ops_downsample < 1:
                    if suggest_level == 0:
                        consider_text = "Consider using level 0."
                    else:
                        consider_text = f"Consider using ops_level<={suggest_level}."
                    raise ValueError(
                        f"Requested tile size at mpp={mpp} "
                        f"on ops_level={ops_level} will require up-sampling. "
                        + consider_text
                    )

        base_downsample = ops_downsample * wsi.properties.level_downsample[ops_level]

        return cls(
            height=tile_h,
            width=tile_w,
            stride_height=stride_h,
            stride_width=stride_w,
            mpp=mpp,
            ops_level=ops_level,
            ops_downsample=ops_downsample,
            base_downsample=base_downsample,
            tissue_name=tissue_name,
        )

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self))

    @cached_property
    def is_overlap_x(self) -> bool:
        """Check if the tiles are overlapped along the x-axis."""
        return self.stride_width < self.width

    @cached_property
    def is_overlap_y(self) -> bool:
        """Check if the tiles are overlapped along the y-axis."""
        return self.stride_height < self.height

    @cached_property
    def is_overlap(self) -> bool:
        """Check if the tiles are overlapped."""
        return self.is_overlap_x or self.is_overlap_y

    @cached_property
    def ops_height(self) -> int:
        return int(self.height * self.ops_downsample)

    @cached_property
    def ops_width(self) -> int:
        return int(self.width * self.ops_downsample)

    @cached_property
    def ops_stride_height(self) -> int:
        return int(self.stride_height * self.ops_downsample)

    @cached_property
    def ops_stride_width(self) -> int:
        return int(self.stride_width * self.ops_downsample)

    @cached_property
    def base_height(self) -> int:
        return int(self.height * self.base_downsample)

    @cached_property
    def base_width(self) -> int:
        return int(self.width * self.base_downsample)

    @cached_property
    def base_stride_height(self) -> int:
        return int(self.stride_height * self.base_downsample)

    @cached_property
    def base_stride_width(self) -> int:
        return int(self.stride_width * self.base_downsample)


def _check_width_height(name, length, default_w=None, default_h=None):
    if length is None:
        if default_w is None or default_h is None:
            raise ValueError(f"{name} cannot be None.")
        w, h = (default_w, default_h)
    elif isinstance(length, Integral):
        w, h = (length, length)
    elif isinstance(length, Sequence):
        w, h = (length[0], length[1])
    else:
        raise TypeError(
            f"Input {name} of {length} is invalid. "
            f"Please use either a tuple of (W, H), or a single integer."
        )
    if not (w > 0 and h > 0):
        raise ValueError(f"{name} must be positive.")
    return w, h
