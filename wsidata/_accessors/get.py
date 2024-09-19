import pandas as pd
from anndata import AnnData


class GetAccessor(object):
    """Accessor for getting information from WSIData object.

    Usage: `wsidata.get`

    """

    def __init__(self, obj):
        self._obj = obj

    def n_tissue(self, key: str) -> int:
        return len(self._obj.sdata.shapes[key])

    def n_tiles(self, key: str) -> int:
        return self.n_tissue(key)

    def pyramids(self) -> pd.DataFrame:
        """Return the pyramid levels of the whole slide image.

        Returns
        -------
        pd.DataFrame
            A table of pyramid levels (index) with columns:
            - height: the height of the level
            - width: the width of the level
            - downsample: the downsample factor of the level

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
    ) -> AnnData:
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
            An AnnData object with the following slots (if present):
            - X: the feature table
            - obs: the data stored in tile table
            - obsm: the spatial information
            - obsp: the spatial graph information
            - uns: include tile_spec and slide_properties

        """

        sdata = self._obj.sdata

        feature_key = self._obj._check_feature_key(feature_key, tile_key)
        feature_adata = sdata.tables[feature_key]
        X = feature_adata.X  # Must be a numpy array

        # layers slot
        layers = feature_adata.layers

        # obs slot
        tile_table = sdata.shapes[tile_key]
        tile_xy = tile_table[["x", "y"]].values
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
        uns = {
            "tile_spec": self._obj.tile_spec(tile_key),
            "slide_properties": self._obj.properties.to_dict(),
        }

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
            X=X, obs=obs, obsm=obsm, obsp=obsp, uns=uns, varm=varm, layers=layers
        )
