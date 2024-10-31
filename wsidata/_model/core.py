from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal, Generator

import numpy as np
from PIL.Image import Image, fromarray
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import SpatialElement

from .tilespec import TileSpec
from .._accessors import FetchAccessor, IterAccessor, DatasetAccessor
from .._utils import find_stack_level
from ..reader import ReaderBase


class WSIData(SpatialData):
    """
    A container class combining :class:`SpatialData <spatialdata.SpatialData>`
    and a whole slide image reader.

    .. note::
     Use the :func:`open_wsi` function to create a WSIData object.

    By default, the whole slide image is not attached to the SpatialData.
    A thumbnail version of the whole slide image is attached for visualization purpose.

    The WSIData contains four main components:

    .. list-table::
     :header-rows: 1

     * -
       - Whole slide image
       - Tissue contours
       - Tile locations
       - Features

     * - **SpatialData Slot**
       - :bdg-danger:`images`
       - :bdg-danger:`shapes`
       - :bdg-danger:`shapes`
       - :bdg-danger:`tables`

     * - **Default Key**
       - :bdg-info:`wsi_thumbnail`
       - :bdg-info:`tissues`
       - :bdg-info:`tiles`
       - :bdg-info:`\{feature_key\}_\{tile_key\}`

     * - | **Attributes**
         | **Slot**
         | **Key**
       - | :class:`SlideProperties <wsidata.reader.SlideProperties>`
         | :bdg-danger:`tables`
         | :bdg-info:`slide_properties`
       -
       - | :class:`TileSpec <wsidata.TileSpec>`
         | :bdg-danger:`tables`
         | :bdg-info:`tile_spec`
       -

     * - **Content**
       - | :class:`DataArray <xarray.DataArray>`
         | (c, y, x) format.
       - | :class:`GeoDataFrame <geopandas.GeoDataFrame>` with columns:
         | :bdg-black:`tissue_id`
         | :bdg-black:`geometry`
       - | :class:`GeoDataFrame <geopandas.GeoDataFrame>` with columns:
         | :bdg-black:`tile_id`
         | :bdg-black:`x`, :bdg-black:`y`
         | :bdg-black:`tissue_id`
         | :bdg-black:`geometry`
       - | :class:`AnnData <anndata.AnnData>` with:
         | :code:`X`: The feature matrix
         | :code:`varm`: :bdg-black:`agg_slide`, :bdg-black:`agg_tissue`



     You can interact with WSIData using the following accessors:

     - :class:`get <wsidata.GetAccessor>`: Access data from the WSIData object.
     - :class:`iter <wsidata.IterAccessor>`: Iterate over data in the WSIData object.
     - :class:`ds <wsidata.DatasetAccessor>`: Create deep learning datasets from the WSIData object.
     - To implement your own accessors, use :func:`register_wsidata_accessor <wsidata.register_wsidata_accessor>`.

     For analysis purpose, you can override two slide properties:

     - microns per pixel (mpp): Using the :meth:`set_mpp` method.
     - bounds: Using the :meth:`set_bounds` method.

     Parameters
     ----------
     reader : :class:`ReaderBase <wsidata.reader.ReaderBase>`
         A reader object that can interface with the whole slide image file.
     sdata : :class:`SpatialData <spatialdata.SpatialData>`
         A SpatialData object for storing analysis data.
     backed_file : str or Path
         Storage location to the SpatialData object.
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
     sdata : :class:`SpatialData <spatialdata.SpatialData>`
         The SpatialData object containing the spatial data.
     backed_file : Path
         The path to the backed file.

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
        )

        if self.SLIDE_PROPERTIES_KEY not in self:
            self.tables[self.SLIDE_PROPERTIES_KEY] = AnnData(
                uns=reader.properties.to_dict()
            )
        else:
            # Try to load the slide properties from the spatial data
            if slide_properties_source == "slide":
                reader_properties = self.tables[self.SLIDE_PROPERTIES_KEY].uns
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
            f"WSI: {self.reader.file}\n"
            f"Reader: {self.reader.name}\n"
            f"{super().__repr__()}"
        )

    def set_exclude_elements(self, elements):
        self._exclude_elements.update(elements)

    def set_wsi_store(self, store: str | Path):
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
    def properties(self):
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
        if self.TILE_SPEC_KEY in self:
            spec = self.tables[self.TILE_SPEC_KEY].uns[key]
            return TileSpec(**spec)

    def set_mpp(self, mpp):
        """Set the microns per pixel (mpp) of the whole slide image."""
        self.properties.mpp = mpp
        self.tables[self.SLIDE_PROPERTIES_KEY].uns["mpp"] = mpp

    def set_bounds(self, bounds):
        """Set the bounds of the whole slide image."""
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

    # def save(self, file=None, consolidate_metadata: bool = True):
    #     # Create the store first
    #     if file is not None:
    #         file = Path(file)
    #         if self.path is None:
    #             self.path = file
    #     else:
    #         file = self._wsi_store
    #     store = parse_url(file, mode="w").store
    #     self.path = file
    #     _ = zarr.group(store=store, overwrite=False)
    #     store.close()
    #     # WARNING: This is not thread-safe, data may be corrupted if multiple threads write to the same file
    #     # Waiting for SpatialData to support thread-safe writing
    #     for elem_type, elem_key, elem in self._gen_elements(include_table=True):
    #         self._write_element(
    #             element=elem,
    #             zarr_container_path=file,
    #             element_type=elem_type,
    #             element_name=elem_key,
    #             overwrite=True,
    #         )
    #     if consolidate_metadata:
    #         self.write_consolidated_metadata()

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
