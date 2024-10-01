from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence, Literal

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from PIL.Image import Image, fromarray
from anndata import AnnData
from ome_zarr.io import parse_url
from shapely import box, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from .tilespec import TileSpec
from .._accessors import GetAccessor, IterAccessor, DatasetAccessor
from .._utils import find_stack_level
from ..reader import ReaderBase


class WSIData(object):
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

    def __init__(
        self,
        reader: ReaderBase,
        sdata: SpatialData,
        backed_file: str | Path,
        slide_properties_source: Literal["slide", "sdata"] = "sdata",
    ):
        self._reader = reader
        self._sdata = sdata
        self.set_backed_file(backed_file)
        self._write_elements = set()

        if self.SLIDE_PROPERTIES_KEY not in sdata:
            sdata.tables[self.SLIDE_PROPERTIES_KEY] = AnnData(
                uns=reader.properties.to_dict()
            )
            self._write_elements.add(self.SLIDE_PROPERTIES_KEY)
        else:
            # Try to load the slide properties from the spatial data
            if slide_properties_source == "slide":
                reader_properties = sdata.tables[self.SLIDE_PROPERTIES_KEY].uns
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
            f"{self.sdata.__repr__()}"
        )

    def __getitem__(self, item):
        return self.sdata.__getitem__(item)

    def close(self):
        """Close the reader object."""
        self.reader.detach_reader()

    def add_write_elements(self, name: str | Sequence[str]):
        """Add an element or elements that need to save to backed file on disk."""
        if isinstance(name, str):
            self._write_elements.add(name)
        else:
            self._write_elements.update(name)

    @property
    def reader(self):
        return self._reader

    @property
    def sdata(self):
        return self._sdata

    @property
    def properties(self):
        return self.reader.properties

    @property
    def backed_file(self):
        return self._backed_file

    @property
    def thumbnail(self):
        return self.get_thumbnail(size=500, as_array=False)

    def tile_spec(self, key: str) -> TileSpec:
        """
        Get the :class:`TileSpec` for a collection of tiles.

        Parameters
        ----------
        key : str
            The key of the tiles.

        """
        if self.TILE_SPEC_KEY in self.sdata:
            spec = self.sdata.tables[self.TILE_SPEC_KEY].uns[key]
            return TileSpec(**spec)

    def set_mpp(self, mpp):
        """Set the microns per pixel (mpp) of the whole slide image."""
        self.properties.mpp = mpp
        self.sdata.tables[self.SLIDE_PROPERTIES_KEY].uns["mpp"] = mpp
        self._write_elements.add(self.SLIDE_PROPERTIES_KEY)

    def set_bounds(self, bounds):
        """Set the bounds of the whole slide image."""
        self.properties.bounds = bounds
        self.sdata.tables[self.SLIDE_PROPERTIES_KEY].uns["bounds"] = bounds
        self._write_elements.add(self.SLIDE_PROPERTIES_KEY)

    def set_backed_file(self, file):
        self._backed_file = Path(file)

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

    def add_tissues(self, key: str, tissues: Sequence[Polygon], ids=None, **kws):
        """Add tissue contours to the SpatialData.

        - :bdg-primary:`SpatialData Slot`: :bdg-danger:`shapes`
        - :bdg-info:`Table Columns`: :bdg-black:`tissue_id`, :bdg-black:`geometry`

        Parameters
        ----------
        key : str
            The key of the tissue contours.
        tissues : array of :class:`Polygon <shapely.Polygon>`
            A sequence of shapely Polygon objects.
        ids : Sequence[int], optional
            The tissue ids.


        """
        if ids is None:
            ids = np.arange(len(tissues))
        gdf = gpd.GeoDataFrame(
            data={
                "tissue_id": ids,
                "geometry": tissues,
            }
        )
        cs = ShapesModel.parse(gdf, **kws)
        self.sdata.shapes[key] = cs
        self._write_elements.add(key)

    def add_shapes(self, key, shapes, **kws):
        cs = ShapesModel.parse(shapes, **kws)
        self.sdata.shapes[key] = cs
        self._write_elements.add(key)

    def add_tiles(self, key, xys, tile_spec, tissue_ids, **kws):
        # Tiles should be stored as polygon
        # This allows easy query of which cells in which tiles
        w, h = tile_spec.raw_width, tile_spec.raw_height
        gdf = gpd.GeoDataFrame(
            data={
                "id": np.arange(len(xys)),
                "x": xys[:, 0],
                "y": xys[:, 1],
                "tissue_id": tissue_ids,
                "geometry": [box(x, y, x + w, y + h) for (x, y) in xys],
            }
        )
        cs = ShapesModel.parse(gdf, **kws)
        self.sdata.shapes[key] = cs

        if self.TILE_SPEC_KEY in self.sdata.tables:
            spec_data = self.sdata.tables[self.TILE_SPEC_KEY]
            spec_data.uns[key] = tile_spec.to_dict()
        else:
            spec_data = ad.AnnData(uns={key: tile_spec.to_dict()})
            self.sdata.tables[self.TILE_SPEC_KEY] = spec_data

        self._write_elements.add(key)
        self._write_elements.add(self.TILE_SPEC_KEY)

    def update_shapes_data(self, key: str, data: Mapping | pd.DataFrame):
        shapes = self.sdata.shapes[key]
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="series")
        for k, v in data.items():
            shapes[k] = v
        self._write_elements.add(key)

    def add_features(self, key, tile_key, features, **kws):
        tile_id = np.arange(len(features))
        library_id = pd.Categorical([tile_key])

        if "obs" in kws:
            obs = kws["obs"]
            obs["tile_id"] = tile_id
            obs["library_id"] = library_id
        else:
            obs = pd.DataFrame(
                {"tile_id": tile_id, "library_id": library_id},
                index=tile_id.astype(str),
            )
            kws["obs"] = obs
        adata = AnnData(X=features, **kws)

        model_kws = dict(
            region=tile_key, region_key="library_id", instance_key="tile_id"
        )
        feature_table = TableModel.parse(adata, **model_kws)
        self.sdata.tables[key] = feature_table
        self._write_elements.add(key)

    def add_agg_features(self, key, agg_key, agg_features, by_key=None, by_data=None):
        feature_table = self.sdata.tables[key]
        feature_table.varm[agg_key] = agg_features

        agg_ops = feature_table.uns.get("agg_ops", {})
        agg_ops[agg_key] = by_key
        feature_table.uns["agg_ops"] = agg_ops
        if by_data is not None:
            feature_table.obs[by_key] = by_data

        self._write_elements.add(key)

    def add_table(self, key, table, **kws):
        table = TableModel.parse(table, **kws)
        self.sdata.tables[key] = table
        self._write_elements.add(key)

    def save(self, file=None, consolidate_metadata: bool = True):
        # Create the store first
        if file is not None:
            self.set_backed_file(file)
        store = parse_url(self._backed_file, mode="w").store
        _ = zarr.group(store=store, overwrite=False)
        store.close()

        # Assign to SpatialData
        self.sdata.path = self._backed_file

        self.sdata.write_element(list(self._write_elements), overwrite=True)
        if consolidate_metadata:
            self.sdata.write_consolidated_metadata()

    def _check_feature_key(self, feature_key, tile_key=None):
        msg = f"{feature_key} doesn't exist"
        if feature_key in self.sdata:
            return feature_key
        else:
            if tile_key is not None:
                feature_key = f"{feature_key}_{tile_key}"
                if feature_key in self.sdata:
                    return feature_key
                msg = f"Neither {feature_key} or {feature_key}_{tile_key} exist"

        raise KeyError(msg)

    # Define default accessors as property,
    # this will enable auto-completion in IDE
    # For extension, user can define their own accessors
    # using register_wsidata_accessor
    @cached_property
    def get(self):
        return GetAccessor(self)

    @cached_property
    def iter(self):
        return IterAccessor(self)

    @cached_property
    def ds(self):
        return DatasetAccessor(self)
