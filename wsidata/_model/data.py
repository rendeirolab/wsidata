from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from ome_zarr.io import parse_url
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

from .tilespec import TileSpec
from .._accessors import GetAccessor, IterAccessor
from .._reader import ReaderBase


class WSIData(object):
    """WSI Data

    WSI data is a container class (SpatialData + Reader)
    to fit the use of whole slide images.

    The wsi data will be initialized with a _reader object that can
    operate the whole slide image file, by default, the whole slide image
    will not be attached to the SpatialData. However, for visualization purpose,
    a thumbnail version of the whole slide image can be attached.

    There are three main components in the SlideData:

    - Whole slide image (WSI)
    - Tissues contours (shapes)
    - Tiles locations (shapes)
    - Features (tables)

    """

    TILE_SPEC_KEY = "tile_spec"
    SLIDE_PROPERTIES_KEY = "slide_properties"

    def __init__(self, reader: ReaderBase, sdata: SpatialData, backed_file: str | Path):
        self._reader = reader
        self._sdata = sdata
        self.set_backed_file(backed_file)
        self._write_elements = set()

        if self.SLIDE_PROPERTIES_KEY not in sdata:
            sdata.tables[self.SLIDE_PROPERTIES_KEY] = AnnData(
                uns=reader.properties.to_dict()
            )
            self._write_elements.add(self.SLIDE_PROPERTIES_KEY)

    def __repr__(self):
        return (
            f"WSI: {self.reader.file}\n"
            f"Reader: {self.reader.name}\n"
            f"{self.sdata.__repr__()}"
        )

    def __getitem__(self, item):
        return self.sdata.__getitem__(item)

    def close(self):
        self.reader.detach_reader()

    def add_write_elements(self, name: str | Sequence[str]):
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

    def n_tissue(self, key: str) -> int:
        return len(self.sdata.shapes[key])

    def n_tiles(self, key: str) -> int:
        return self.n_tissue(key)

    def tile_spec(self, key: str) -> TileSpec:
        if self.TILE_SPEC_KEY in self.sdata:
            spec = self.sdata.tables[self.TILE_SPEC_KEY].uns[key]
            return TileSpec(**spec)

    def set_mpp(self, mpp):
        # TODO: Allow user to set the mpp of slide
        pass

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
        return self.reader.get_region(x, y, width, height, level=level, **kwargs)

    def add_tissues(self, key: str, tissues: Sequence[Polygon], ids=None, **kws):
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
                "geometry": [
                    Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
                    for (x, y) in xys
                ],
            }
        )
        cs = ShapesModel.parse(gdf, **kws)
        self.sdata.shapes[key] = cs

        if self.TILE_SPEC_KEY in self.sdata.tables:
            spec_data = self.sdata.tables[self.TILE_SPEC_KEY]
        else:
            spec_data = ad.AnnData()
        spec_data.uns[key] = tile_spec.to_dict()
        self.sdata.tables[self.TILE_SPEC_KEY] = spec_data

        self._write_elements.add(key)
        self._write_elements.add(self.TILE_SPEC_KEY)

    def update_shapes_data(self, key: str, data: Mapping | pd.DataFrame):
        shapes = self.sdata.shapes[key]
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="series")
        self.sdata.shapes[key] = shapes.assign(**data)
        self._write_elements.add(key)

    def add_features(self, key, tile_key, features):
        tile_id = np.arange(len(features))
        obs = pd.DataFrame(
            {"tile_id": tile_id, "library_id": tile_key},
            index=tile_id.astype(str),
        )
        adata = AnnData(X=features, obs=obs)

        feature_table = TableModel.parse(
            adata, region=tile_key, region_key="library_id", instance_key="tile_id"
        )
        self.sdata.tables[key] = feature_table
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
