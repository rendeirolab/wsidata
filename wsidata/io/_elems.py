from __future__ import annotations

__all__ = [
    "add_tissues",
    "add_shapes",
    "add_tiles",
    "add_features",
    "add_agg_features",
    "add_table",
    "update_shapes_data",
    "subset_tiles",
    "sync_tile_spec",
]

from typing import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from shapely import Polygon, box
from spatialdata.models import ShapesModel, TableModel

from .._model import TileSpec, WSIData


def add_tissues(
    wsidata: WSIData, key: str, tissues: Sequence[Polygon], ids=None, **kws
):
    """Add tissue contours to the SpatialData.

    - :bdg-primary:`SpatialData Slot`: :bdg-danger:`shapes`
    - :bdg-info:`Table Columns`: :bdg-black:`tissue_id`, :bdg-black:`geometry`

    Parameters
    ----------
    wsidata : WSIData
        The WSIData object.
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
    wsidata.shapes[key] = cs


def add_tiles(wsidata, key, xys, tile_spec: TileSpec, tissue_ids, **kws):
    # Tiles should be stored as polygon
    # This allows easy query of which cells in which tiles
    w, h = tile_spec.base_width, tile_spec.base_height
    gdf = gpd.GeoDataFrame(
        data={
            "tile_id": np.arange(len(xys)),
            "x": xys[:, 0],
            "y": xys[:, 1],
            "tissue_id": tissue_ids,
            "geometry": [box(x, y, x + w, y + h) for (x, y) in xys],
        }
    )
    cs = ShapesModel.parse(gdf, **kws)
    wsidata.shapes[key] = cs

    if wsidata.TILE_SPEC_KEY in wsidata.attrs:
        spec_data = wsidata.attrs[wsidata.TILE_SPEC_KEY]
        spec_data[key] = tile_spec.to_dict()
    else:
        spec_data = {key: tile_spec.to_dict()}
        wsidata.attrs[wsidata.TILE_SPEC_KEY] = spec_data


def subset_tiles(wsidata, key, indices, new_key=None):
    shapes = wsidata.shapes[key]
    shapes = shapes.iloc[indices]
    if new_key is None:
        new_key = key
    else:
        spec_data = wsidata.attrs[wsidata.TILE_SPEC_KEY]
        spec_data[new_key] = spec_data[key]
    wsidata.shapes[new_key] = shapes


def add_features(wsidata, key, tile_key, features, **kws):
    tile_tb = wsidata[tile_key]
    if "tile_id" in tile_tb.columns:
        tile_id = tile_tb["tile_id"].values
    else:
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

    model_kws = dict(region=tile_key, region_key="library_id", instance_key="tile_id")
    feature_table = TableModel.parse(adata, **model_kws)
    wsidata.tables[key] = feature_table


def add_agg_features(wsidata, key, agg_key, agg_features, by_key=None, by_data=None):
    feature_table = wsidata.tables[key]
    feature_table.varm[agg_key] = agg_features

    agg_ops = feature_table.uns.get("agg_ops", {})
    agg_ops[agg_key] = by_key
    feature_table.uns["agg_ops"] = agg_ops
    if by_data is not None:
        feature_table.obs[by_key] = by_data


# The following might be unnecessary? Should we remove it?
def add_shapes(wsidata, key, shapes, **kws):
    cs = ShapesModel.parse(shapes, **kws)
    wsidata.shapes[key] = cs


def add_table(wsidata, key, table, **kws):
    table = TableModel.parse(table, **kws)
    wsidata.tables[key] = table


def update_shapes_data(wsidata, key: str, data: Mapping | pd.DataFrame):
    shapes = wsidata.shapes[key]
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="series")
    for k, v in data.items():
        shapes[k] = v


def sync_tile_spec(wsidata, from_key, to_key):
    """
    Sync `TileSpec` between tile tables

    Parameters
    ----------
    wsidata : :class:`wsidata.WSIData`
        The wsidata to sync
    from_key : str
        The key of the tile table to be synced
    to_key : str
        The key of the tile table to sync with

    """
    spec_data = wsidata.attrs[wsidata.TILE_SPEC_KEY]
    spec_data[to_key] = spec_data[from_key]
