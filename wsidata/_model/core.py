from __future__ import annotations

import base64
import io
import json
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from numbers import Integral, Number
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, Sequence

import numpy as np
from anndata import AnnData
from ome_zarr.io import parse_url
from PIL.Image import Image, fromarray
from spatialdata import SpatialData
from spatialdata.models import SpatialElement

from .._utils import find_stack_level
from ..accessors import DatasetAccessor, FetchAccessor, IterAccessor
from ..reader import ReaderBase, SlideProperties

if TYPE_CHECKING:
    from typing import Self

    from wsidata.reader.base import AssociatedImages


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
         - :class:`SlideProperties <wsidata.SlideProperties>`

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
    properties : :class:`SlideProperties <wsidata.SlideProperties>`
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
            attrs=sdata.attrs,
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

    def __repr_texts(self):
        H, W = self.properties.shape
        dimension_text = f"{H}×{W} (h×w)"
        n_level = self.properties.n_level
        pyramid_text = f"{n_level} {'Pyramid' if n_level == 1 else 'Pyramids'}"
        mpp_text = "Unknown"
        if self.properties.mpp is not None:
            mpp_text = f"{self.properties.mpp:.2f} MPP"
        if self.properties.magnification is not None:
            mpp_text += f" ({int(self.properties.magnification)}X)"
        return dimension_text, pyramid_text, mpp_text

    def __repr__(self):
        dimension_text, pyramid_text, mpp_text = self.__repr_texts()

        return (
            f"WSI: {self.reader.file}\n"
            f"Reader: {self.reader.name}\n"
            f"Dimensions: {dimension_text}, {pyramid_text}\n"
            f"Pixel physical size: {self.properties.mpp} MPP\n"
            f"{super().__repr__()}"
        )

    def _repr_html_(self):
        spatialdata_repr = super().__repr__()
        dimension_text, pyramid_text, mpp_text = self.__repr_texts()

        return f"""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="data:image/png;base64,{self._html_thumbnail}" 
                    style="border: 1px solid #ddd; border-radius: 8px; 
                    max-width: 300px; max-height: 100%; object-fit: contain;">
                    <div>
                        <b>WSI:</b> {self.reader.file}<br>
                        <b>Reader:</b> {self.reader.name}<br>
                        <b>Dimensions:</b> {dimension_text}, {pyramid_text}<br>
                        <b>Pixel physical size:</b> {mpp_text}<br>
                        <pre style="padding: 8px; border-radius: 4px; font-size: 8pt">{spatialdata_repr}</pre>
                    </div>
                </div>"""

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
    def reader(self) -> ReaderBase:
        """The reader object for interfacing with the whole slide image."""
        return self._reader

    @property
    def properties(self) -> SlideProperties:
        """The properties of the whole slide image. See :class:`SlideProperties <wsidata.SlideProperties>`."""
        return self.reader.properties

    @property
    def raw_properties(self) -> dict:
        """The raw properties of the whole slide image."""
        return self.reader.raw_properties

    @property
    def associated_images(self) -> AssociatedImages:
        """The associated images in a key-value pair."""
        return self.reader.associated_images

    @property
    def wsi_store(self) -> str | Path:
        """The zarr store path for the associated data of the whole slide image."""
        return self._wsi_store

    @property
    def thumbnail(self) -> Image:
        """The thumbnail of the whole slide image."""
        return self.get_thumbnail(size=500, as_array=False)

    @cached_property
    def _html_thumbnail(self) -> str:
        buffer = io.BytesIO()
        image = self.get_thumbnail(size=250, as_array=False)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @property
    def name(self) -> str:
        """The file name of the whole slide image."""
        return Path(self.reader.file).name

    def tile_spec(self, key: str) -> TileSpec | None:
        """
        Get the :class:`TileSpec` for a collection of tiles.

        Parameters
        ----------
        key : str
            The key of the tiles.

        """
        if self.TILE_SPEC_KEY in self.attrs:
            spec = self.attrs[self.TILE_SPEC_KEY].get(key)
            if spec is not None:
                return TileSpec(**spec)

    def set_mpp(self, mpp):
        """Set the microns per pixel (mpp) of the whole slide image.
        This will override the recorded mpp in the slide properties.
        Could be useful when the mpp is not recorded in the image file.

        Parameters
        ----------
        mpp : float
            The microns per pixel.

        """
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
            if self._wsi_store is None and self.path is None:
                raise ValueError(
                    "The store path for the WSIData is not set. "
                    "Please set the store path before saving."
                )
            file_path = self._wsi_store
        super().write(
            file_path=file_path,
            overwrite=overwrite,
            consolidate_metadata=consolidate_metadata,
            format=format,
        )

    def to_spatialdata(self) -> SpatialData:
        """
        Convert the WSIData object to a SpatialData object.

        .. note::
            This is not a deep copy operation.
            Any changes to the returned SpatialData will affect the original WSIData.

        """
        return SpatialData(
            images=self.images,  # noqa
            labels=self.labels,  # noqa
            shapes=self.shapes,  # noqa
            tables=self.tables,
            points=self.points,  # noqa
            attrs=self.attrs,
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

    def __repr__(self):
        return (
            f"Tile at {self.mpp} mpp, {self.height}×{self.width} (h×w)\n"
            f"Stride: {self.stride_height}×{self.stride_width} ({self.overlap_y}×{self.overlap_x} overlap)\n"
            f"Operation size: {self.ops_height}×{self.ops_width}, level={self.ops_level}\n"
            f"Base size: {self.base_height}×{self.base_width}, level={self.base_level}\n"
            f"Target tissue: '{self.tissue_name}'"
        )

    def _repr_html_(self):
        BORDER_WIDTH = 2
        scale = 100 / max(self.width, self.height)
        scaled_w = np.floor(self.width * scale) - BORDER_WIDTH
        scaled_h = np.floor(self.height * scale) - BORDER_WIDTH
        scaled_stride_w = np.floor(
            self.stride_width * scale
        )  # add 2 to account for the border
        scaled_stride_h = np.floor(
            self.stride_height * scale
        )  # add 2 to account for the border

        tile_style = (
            f"width: {int(scaled_w)}px; "
            f"height: {int(scaled_h)}px; "
            f"position: absolute; "
            f"box-sizing: content-box;"  # make sure the border is included in the size
            f"border: {BORDER_WIDTH}px solid #000; "
        )
        container_style = (
            f"width: {int(scaled_w + scaled_stride_w) + 2 * BORDER_WIDTH}px; "
            f"height: {int(scaled_h + scaled_stride_h) + 2 * BORDER_WIDTH}px; "
            f"position: relative;"
            f"border: 0;"
        )
        html = f"""
        <div style="display: flex; align-item: center; gap: 10px;">
            <div style='{container_style}'>
                <div style='{tile_style}; top: 0px; left: 0px; border-color: #C68FE6'>
                    <p style='padding-left: 3pt'>Tile 1</p>
                </div>
                <div style='{tile_style}; top: 0px; left: {scaled_stride_w}px; border-color: #F6DC43'>
                    <p style='padding-left: 3pt'>Tile 2</p>
                </div>
                <div style='{tile_style}; top: {scaled_stride_w}px; left: 0px; border-color: #6A9C89'>
                    <p style='padding-left: 3pt'>Tile 3</p>
                </div>
            </div>
            <div>
                <b>Tile at</b>: {self.mpp} mpp<br>
                <b>Tile size</b>: {self.height}×{self.width} (h×w)<br>
                <b>Stride</b>: {self.stride_height}×{self.stride_width} ({self.overlap_y}×{self.overlap_x} overlap)<br>
                <b>Operation size</b>: {self.ops_height}×{self.ops_width}, level={self.ops_level}<br>
                <b>Base size</b>: {self.base_height}×{self.base_width}, level={self.base_level}<br>
                <b>Target tissue</b>: '{self.tissue_name}'
            </div>
        </div>
        """
        return html

    @classmethod
    def from_wsidata(
        cls,
        wsi: WSIData,
        tile_px: int | (int, int),
        stride_px: int | (int, int) = None,
        overlap: float | (float, float) = None,
        mpp=None,
        ops_level=None,
        slide_mpp=None,
        tissue_name=None,
    ) -> Self:
        """Create a TileSpec from a WSIData object.

        To tile from the whole slide image, the user needs to specify the tile size and stride size.
        mpp only need to be specified if the user wants to make sure
        the tile size is harmonized across different slides.

        If ops_level is not specified, the optimal level will be calculated based on the requested mpp
        to maximize the performance.

        """
        # Check if the input params are valid
        tile_w, tile_h, stride_w, stride_h = _preprocess_tile_params(
            tile_px, stride_px, overlap
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
        """Convert the TileSpec to a dictionary."""
        return asdict(self)

    def to_json(self):
        """Convert the TileSpec to a JSON string."""
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
    def overlap_x(self) -> int:
        """The overlap pixel size along the x-axis."""
        return self.width - self.stride_width

    @cached_property
    def overlap_y(self) -> int:
        """The overlap pixel size along the y-axis."""
        return self.height - self.stride_height

    @cached_property
    def ops_height(self) -> int:
        """The height of the tile when retrieving images."""
        return int(self.height * self.ops_downsample)

    @cached_property
    def ops_width(self) -> int:
        """The width of the tile when retrieving images."""
        return int(self.width * self.ops_downsample)

    @cached_property
    def ops_stride_height(self) -> int:
        """The height of the stride when retrieving images."""
        return int(self.stride_height * self.ops_downsample)

    @cached_property
    def ops_stride_width(self) -> int:
        """The width of the stride when retrieving images."""
        return int(self.stride_width * self.ops_downsample)

    @cached_property
    def base_height(self) -> int:
        """The height of the tile at the level 0."""
        return int(self.height * self.base_downsample)

    @cached_property
    def base_width(self) -> int:
        """The width of the tile at the level 0."""
        return int(self.width * self.base_downsample)

    @cached_property
    def base_stride_height(self) -> int:
        """The height of the stride at the level 0."""
        return int(self.stride_height * self.base_downsample)

    @cached_property
    def base_stride_width(self) -> int:
        """The width of the stride at the level 0."""
        return int(self.stride_width * self.base_downsample)


def _preprocess_tile_params(
    tile_px: int | (int, int),
    stride_px: int | (int, int) = None,
    overlap: Number | (Number, Number) = 0.5,
):
    if isinstance(tile_px, Integral):
        tile_w, tile_h = (tile_px, tile_px)
    elif isinstance(tile_px, Sequence):
        tile_w, tile_h = (tile_px[0], tile_px[1])
    else:
        raise TypeError("tile_px must be an integer or a tuple of two integers.")

    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile_px must be positive.")

    if stride_px is not None and overlap is not None:
        raise ValueError("Cannot specify both stride_px and overlap.")

    stride_w, stride_h = tile_w, tile_h
    if stride_px is not None:
        if isinstance(stride_px, Integral):
            stride_w, stride_h = (stride_px, stride_px)
        elif isinstance(stride_px, Sequence):
            stride_w, stride_h = (stride_px[0], stride_px[1])
        else:
            raise TypeError("stride_px must be an integer or a tuple of two integers.")
        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("stride_px must be positive.")

    if overlap is not None:
        if isinstance(overlap, Number):
            overlap_w, overlap_h = (overlap, overlap)
        elif isinstance(overlap, Sequence):
            overlap_w, overlap_h = (overlap[0], overlap[1])
        else:
            raise TypeError("overlap must be a number or a tuple of two numbers.")

        if overlap_w < 0 or overlap_h < 0:
            raise ValueError("overlap must be non-negative.")
        # calculate stride px from overlap

        if overlap_w < 1:
            stride_w = int(tile_w * (1 - overlap_w))
        else:
            stride_w = int(tile_w - overlap_w)

        if overlap_h < 1:
            stride_h = int(tile_h * (1 - overlap_h))
        else:
            stride_h = int(tile_h - overlap_h)

    return tile_w, tile_h, stride_w, stride_h
