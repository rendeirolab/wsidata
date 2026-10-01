from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pandas as pd


def _tile_rows(sdata, feature_key, tile_key):
    """The row in the feature table of each tile, in the order of the tiles.

    Features link to their tiles by tile_id. Without a tile_id in both
    tables, they can only be matched to the tiles by position.
    """
    features = sdata.tables[feature_key]
    tiles = sdata.shapes[tile_key]
    if "tile_id" not in tiles.columns or "tile_id" not in features.obs.columns:
        if len(tiles) != features.n_obs:
            raise ValueError(
                f"'{tile_key}' has {len(tiles)} rows and '{feature_key}' has "
                f"{features.n_obs}. Without a tile_id in both, features can "
                "only be matched to the tiles by position."
            )
        return np.arange(features.n_obs)
    rows = pd.Index(features.obs["tile_id"]).get_indexer(tiles["tile_id"])
    missing = tiles["tile_id"].to_numpy()[rows == -1]
    if missing.size > 0:
        raise ValueError(
            f"No features in '{feature_key}' for {missing.size} of the "
            f"{len(tiles)} tiles in '{tile_key}', e.g. tile_id "
            f"{missing[:5].tolist()}. Extract the features of the tiles again."
        )
    return rows


class FetchAccessor(object):
    """Accessor for getting information from WSIData object.

    Usage: `wsidata.fetch`

    """

    def __init__(self, obj):
        self._obj = obj

    def n_tissue(self, key: str) -> int:
        """
        Return the number of tissue regions in the tissue table.

        Parameters
        ----------
        key: str
            The tile key.

        Returns
        -------
        int
            The number of tissue regions.

        """
        return len(self._obj.shapes[key])

    def n_tiles(self, key: str) -> int:
        """
        Return the number of tiles in the tile table.

        Parameters
        ----------
        key: str
            The tile key.

        Returns
        -------
        int
            The number of tiles.
        """
        return self.n_tissue(key)

    def pyramids(self) -> pd.DataFrame:
        """
        Return the pyramid levels of the whole slide image.

        Returns
        -------
        pd.DataFrame
            A table of pyramid levels (index) with columns:

            - height : The height of the level (px).
            - width : The width of the level (px).
            - downsample : The downsample factor of the level.

        """
        heights, widths = zip(*self._obj.properties.level_shape)
        return pd.DataFrame(
            {
                "height": pd.Series(heights, dtype=int),
                "width": pd.Series(widths, dtype=int),
                "downsample": self._obj.properties.level_downsample,
            },
            index=pd.RangeIndex(self._obj.properties.n_level, name="level"),
        )

    def features_anndata(
        self, feature_key, tile_key="tiles", tile_graph=True
    ) -> "AnnData":
        """Return the feature table as an AnnData object.

        Parameters
        ----------
        feature_key : str
            The feature key.
        tile_key : str, default: "tiles"
            The tile key.
        tile_graph : bool, default: True
            If True, include spatial graph information.

        Returns
        -------
        AnnData
            An AnnData object with the following components (if present):

            - X : The features of each tile, in the order of the tile table.
            - obs : The data stored in the tile table.
            - obsm : The x,y coordinates for each tile.
            - obsp : The spatial graph information.
            - uns : Metadata including tile specifications and slide properties.

        """
        from anndata import AnnData

        sdata = self._obj
        feature_key = self._obj._check_feature_key(feature_key, tile_key)
        feature_adata = sdata.tables[feature_key]
        # Rows follow the tiles, like obs, obsm and the tile graph
        rows = _tile_rows(sdata, feature_key, tile_key)
        X = feature_adata.X[rows]  # Must be a numpy array
        var = feature_adata.var

        # layers slot
        # anndata 0.13 also lists X, as layers[None]
        layers = {
            key: layer[rows]
            for key, layer in feature_adata.layers.items()
            if key is not None
        }

        # obs slot
        tile_table = sdata.shapes[tile_key]
        tile_xy = tile_table.bounds[["minx", "miny"]].to_numpy()
        obs = tile_table.drop(columns=["geometry"])
        # To suppress anndata warning
        obs.index = obs.index.astype(str)

        # obsm slot
        obsm = {"spatial": tile_xy}

        # obsp slot
        obsp = {}

        # varm slot
        varm = feature_adata.varm

        # uns slot
        uns = {"slide_properties": self._obj.properties.to_dict()}
        # Shapes other than tiles, such as cells, have no TileSpec
        tile_spec = self._obj.tile_spec(tile_key)
        if tile_spec is not None:
            uns["tile_spec"] = tile_spec.to_dict()

        if tile_graph:
            conns_key = "spatial_connectivities"
            dists_key = "spatial_distances"
            graph_key = f"{tile_key}_graph"
            if graph_key in sdata:
                graph_table = sdata.tables[graph_key]
                obsp[conns_key] = graph_table.obsp[conns_key]
                obsp[dists_key] = graph_table.obsp[dists_key]
                uns["spatial"] = graph_table.uns["spatial"]

        return AnnData(
            X=X,
            layers=layers,
            var=var,
            varm=varm,
            obs=obs,
            obsm=obsm,
            obsp=obsp,
            uns=uns,
        )
