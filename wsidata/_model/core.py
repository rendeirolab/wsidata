from __future__ import annotations

import base64
import io
import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal

import numpy as np
from anndata import AnnData
from ome_zarr.io import parse_url
from PIL.Image import Image, fromarray
from spatialdata import SpatialData
from spatialdata.models import SpatialElement

from .._utils import find_stack_level
from ..accessors import DatasetAccessor, FetchAccessor, IterAccessor
from ..reader import ReaderBase, SlideProperties
from .tile import TileSpec

if TYPE_CHECKING:
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
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <img src="data:image/png;base64,{self._html_thumbnail}"
                    style="border: 1px solid #ddd; border-radius: 8px;
                    max-width: 300px; height: auto; flex-shrink: 0;">
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
        self, include_tables: bool = False
    ) -> Generator[tuple[str, str, SpatialElement | AnnData], None, None]:
        for i in super()._gen_elements(include_tables):
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
